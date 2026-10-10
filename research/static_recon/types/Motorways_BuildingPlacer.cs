using System;
using System.Collections.Generic;
using System.Diagnostics;
using Factory;
using Factory.Pools;
using FixMath;
using Motorways.Models;
using Motorways.Processes;
using Motorways.Views;
using Server;
using UnityEngine;

namespace Motorways;

public class BuildingPlacer : IReusable, IReleasedFromScopeHandler
{
	public class Driveway
	{
		public Vector2Int coordinatesOffset;

		public TileDirection direction;

		public override string ToString()
		{
			return $"Driveway [{coordinatesOffset}, {direction}]";
		}
	}

	public class RailPlatform
	{
		public Vector2Int coordinatesOffset;

		public TileDirectionBitfield connection;

		public override string ToString()
		{
			return $"RailPlatform [{coordinatesOffset}, {connection}]";
		}
	}

	public class Layout
	{
		public Vector2Int footprint;

		public List<Driveway> driveways = new List<Driveway>();

		public List<RailPlatform> platforms = new List<RailPlatform>();

		public List<Vector2Int> boatTerminalTiles = new List<Vector2Int>();

		public TileDirection carparkSide = TileDirection.None;

		public override string ToString()
		{
			string text = "";
			foreach (Driveway driveway in driveways)
			{
				text += $"{driveway}";
			}
			string text2 = "";
			foreach (RailPlatform platform in platforms)
			{
				text2 += $"{platform}";
			}
			return $"Layout [{footprint}, {text}, {text2}, {carparkSide}]";
		}
	}

	public class Placement
	{
		public Vector2Int coordinates;

		public Layout layout;

		public Fix64 weight;

		public override string ToString()
		{
			return $"Placement [{coordinates}, {layout}, {weight}]";
		}
	}

	public enum WeightSource
	{
		Default,
		Station,
		BoatTerminal
	}

	public enum WeightEvaluationLevel
	{
		ExclusivelyUseWeightedTiles,
		AllowNonWeightedTiles,
		IgnoreWeights
	}

	[Dependency]
	private ClockModel _clock;

	[Dependency]
	private City _city;

	[Dependency]
	private CityModel _cityModel;

	[Dependency]
	private CityPlanModel _cityPlanModel;

	[Dependency]
	private TilemapModel _tilemapModel;

	[Dependency]
	private TilePathfinder _pathfinder;

	[Dependency]
	private Pathfinder _lanePathfinder;

	[Dependency]
	private ISimulation _simulation;

	[Dependency]
	private IScope _scope;

	[Dependency]
	private CitySpawningView _spawningView;

	[Dependency]
	private GameBehaviourModel _behaviour;

	[Dependency]
	private SimulationConstantsData _constants;

	private RectInt _placeableArea;

	private readonly List<Tile> _placeableTiles = new List<Tile>();

	private readonly List<Fix64> _placeableTileWeights = new List<Fix64>();

	private readonly List<string> _placeableTileWeightsContext = new List<string>();

	private readonly List<bool> _placeableTileRails = new List<bool>();

	private readonly HashSet<LaneModel> _cachedHouseLanes = new HashSet<LaneModel>();

	private readonly List<bool> _placeableTileDriveabilities = new List<bool>();

	private TileContentType _buildingType;

	private int _groupIndex;

	private GroupingStyle _grouping;

	private Tile _testTile;

	private readonly List<Placement> _possiblePlacements = new List<Placement>();

	[Unscrubbed]
	private readonly List<Placement> _placementPool = new List<Placement>(100);

	private const int PlacementsToAllocate = 100;

	private int _usedPlacementCount;

	private CitySpawningLayerData _baseTileData;

	public const int GroupingStyleRadius = 7;

	public const int DemandGroupingStyleRadius = 12;

	public const int DestinationDeadzoneRadius = 5;

	public static readonly Fix64 SpawnPushFactorVeryStrong = (Fix64)0.425;

	public static readonly Fix64 SpawnPushFactorStrong = (Fix64)0.6;

	public static readonly Fix64 SpawnPushFactorWeak = (Fix64)0.9;

	public static readonly Fix64 SpawnPullFactor = (Fix64)1.3;

	public const int SuburbRadius = 5;

	public static readonly Fix64 PullFactorPerNeighbourDecrease = (Fix64)0.035;

	public static readonly Fix64 PullFactorMinimum = (Fix64)1.05;

	private const int MinimumProjectedDistanceToPathTo = 5;

	public BuildingPlacer()
	{
		while (_placementPool.Count < 100)
		{
			_placementPool.Add(new Placement());
		}
	}

	public void Reset()
	{
		_baseTileData = null;
		_placeableArea = default(RectInt);
		_buildingType = TileContentType.None;
		_groupIndex = 0;
		_grouping = GroupingStyle.Normal;
		_possiblePlacements.Clear();
		_usedPlacementCount = 0;
		_placeableTiles.Clear();
		_placeableTileWeights.Clear();
		_placeableTileDriveabilities.Clear();
		_placeableTileWeightsContext.Clear();
		_placeableTileRails.Clear();
	}

	public void SetTileData(CitySpawningLayerData tileData)
	{
		_baseTileData = tileData;
	}

	private bool IsTileBuildable(Vector2Int tileCoordinates, bool ignoreUnzoneableTiles = false)
	{
		bool flag = !_tilemapModel.IsTileReserved(tileCoordinates) && (_city.Definition.TileIsZoneable(tileCoordinates) || ignoreUnzoneableTiles);
		TileModel tileModel = _tilemapModel.GetTileModel(tileCoordinates);
		if (tileModel != null && flag)
		{
			flag &= tileModel.Tile.CanSetContentType(_buildingType);
			if (_city.Rules.CanBuildingsDemolishUnusedRoads && !flag)
			{
				flag = !tileModel.Tile.AnyRoadHasPermanenceBelowValue(_constants.PercentageOfPermanenceTimerWhereRoadsCannotBeDemolished, RoadState.Live);
				_testTile = CreateDemolishedTestTileFrom(tileModel.Tile);
				flag &= _testTile.CanSetContentType(_buildingType);
				flag &= !IsTileConnectedToBuildingAndHouse(tileModel);
			}
		}
		return flag;
	}

	public void StartPlacing(TileContentType buildingType, int groupIndex, GroupingStyle grouping, WeightEvaluationLevel weightEvaluationLevel = WeightEvaluationLevel.ExclusivelyUseWeightedTiles, WeightSource weightSource = WeightSource.Default)
	{
		_possiblePlacements.Clear();
		_usedPlacementCount = 0;
		_buildingType = buildingType;
		_placeableTiles.Clear();
		_placeableTileWeights.Clear();
		_placeableTileDriveabilities.Clear();
		_placeableTileWeightsContext.Clear();
		_cachedHouseLanes.Clear();
		_placeableTileRails.Clear();
		_groupIndex = groupIndex;
		_grouping = grouping;
		if (_city.Rules.DoesIgnorePlayableArea())
		{
			_placeableArea.SetMinMax(Vector2Int.zero, Vector2Int.zero);
		}
		else
		{
			Fix64 expansionTime = _clock.ExpansionTime;
			RectFixed simulationPlayableAreaAtTime = _city.GetSimulationPlayableAreaAtTime(expansionTime, City.PlayableAreaRoundingType.ForceWholeTiles);
			Vector2Int vector2Int = new Vector2Int((int)(long)simulationPlayableAreaAtTime.xMin, (int)(long)simulationPlayableAreaAtTime.yMin);
			if (_city.Rules.AllowSpawningAtMapEdges)
			{
				_placeableArea.SetMinMax(vector2Int, new Vector2Int((int)(long)simulationPlayableAreaAtTime.xMax + 1, (int)(long)simulationPlayableAreaAtTime.yMax + 1));
			}
			else
			{
				_placeableArea.SetMinMax(vector2Int + Vector2Int.one, new Vector2Int((int)(long)simulationPlayableAreaAtTime.xMax, (int)(long)simulationPlayableAreaAtTime.yMax));
			}
			int key = CityTilemap.LayerIdFor((_buildingType == TileContentType.House) ? CityTileType.Supply : CityTileType.Demand, groupIndex);
			BuildingSpawningTileWeights value = null;
			if (weightEvaluationLevel != WeightEvaluationLevel.IgnoreWeights)
			{
				switch (weightSource)
				{
				case WeightSource.Station:
					if (Diagnostics.Verify(_baseTileData?.stationWeights != null, "Trying to place station but there are no station weights available."))
					{
						value = _baseTileData.stationWeights;
					}
					break;
				case WeightSource.BoatTerminal:
					if (Diagnostics.Verify(_baseTileData?.boatTerminalWeights != null, "Trying to place ferry terminal but there are no ferry terminal weights available."))
					{
						value = _baseTileData.boatTerminalWeights;
					}
					break;
				case WeightSource.Default:
					Diagnostics.Verify(_baseTileData != null && _baseTileData.weights.TryGetValue(key, out value), "There is no layer present for {0} {1}!", _buildingType, groupIndex);
					break;
				default:
					throw new ArgumentOutOfRangeException("weightSource", weightSource, null);
				}
			}
			for (int i = 0; i < _placeableArea.height; i++)
			{
				for (int j = 0; j < _placeableArea.width; j++)
				{
					Vector2Int vector2Int2 = new Vector2Int(_placeableArea.xMin + j, _placeableArea.yMin + i);
					bool flag = _city.Definition.TileIsDriveable(vector2Int2);
					bool item = _city.Definition.TileIsOverRail(vector2Int2);
					bool num = flag && IsTileBuildable(vector2Int2, _city.Rules.AllowPlacingBuildingsOnUnzoneableTiles);
					Tile tile = _tilemapModel.GetTile(vector2Int2);
					_placeableTiles.Add(tile);
					_placeableTileRails.Add(item);
					if (num)
					{
						Fix64 fix = GetBaseWeightForTile(vector2Int2, weightEvaluationLevel, value);
						if (fix > Fix64.Zero && _behaviour.BuildingSpawnsAreAffectedByOtherBuildings())
						{
							fix = ScaleTileWeightByBuildingInfluence(fix, groupIndex, vector2Int2, buildingType, _placeableTileWeightsContext.Count - 1);
						}
						_placeableTileWeights.Add((fix >= Fix64.Zero) ? fix : (-Fix64.One));
					}
					else
					{
						_placeableTileWeights.Add(-Fix64.One);
					}
					bool flag2 = tile == null || tile.ContentType == TileContentType.None || (tile.ContentType == TileContentType.Tree && _city.Rules.ShouldBuildingsBulldozeTrees);
					_placeableTileDriveabilities.Add(flag && flag2);
				}
			}
			TilemapModel model = _simulation.GetModel<TilemapModel>();
			foreach (Vector2Int allTileCoordinate in model.GetAllTileCoordinates())
			{
				if (!model.GetTile(allTileCoordinate).IsCenterOfRoundabout)
				{
					continue;
				}
				for (int k = -1; k <= 1; k++)
				{
					for (int l = -1; l <= 1; l++)
					{
						if (Math.Abs(k) != 1 || Math.Abs(l) != 1 || !_city.Rules.AllowSpawnsOnRoundaboutDeadzone)
						{
							Vector2Int vector2Int3 = allTileCoordinate + new Vector2Int(k, l);
							if (_placeableArea.Contains(vector2Int3))
							{
								Vector2Int vector2Int4 = vector2Int3 - _placeableArea.min;
								int index = vector2Int4.x + vector2Int4.y * _placeableArea.width;
								_placeableTileWeights[index] = -Fix64.One;
							}
						}
					}
				}
			}
		}
		ModelListEnumerator<HouseModel> enumerator2 = _simulation.GetModels<HouseModel>().GetEnumerator();
		while (enumerator2.MoveNext())
		{
			HouseModel current2 = enumerator2.Current;
			Vector2Int coordinates = current2.tileModel.Coordinates;
			DensityGroup densityGroup = _city.Definition.DensityForPosition(coordinates);
			int radius = (int)densityGroup * 2;
			string context = "";
			switch (densityGroup)
			{
			case DensityGroup.Low:
				context = "low density tile";
				break;
			case DensityGroup.Medium:
				context = "medium density tile";
				break;
			case DensityGroup.High:
				context = "high density tile";
				break;
			}
			ChangeTileWeightsAroundBuilding(coordinates, radius, Fix64Consts.OneHalf, BuildingSpawningProcess.HouseFootprint, context);
			if (!_behaviour.UseDestinationDeadzonesFor(CityTileType.Supply) || (buildingType != TileContentType.Destination && buildingType != TileContentType.Carpark))
			{
				continue;
			}
			foreach (Vector2Int item2 in TileUtilities.GetBoundsWithBoundary(current2.tileModel.Coordinates, BuildingSpawningProcess.HouseFootprint).allPositionsWithin)
			{
				if (_placeableArea.Contains(item2))
				{
					Vector2Int vector2Int5 = item2 - _placeableArea.min;
					int index2 = vector2Int5.x + vector2Int5.y * _placeableArea.width;
					_placeableTileWeights[index2] = -Fix64.One;
				}
			}
		}
		ModelListEnumerator<DestinationModel> enumerator3 = _simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator3.MoveNext())
		{
			DestinationModel current4 = enumerator3.Current;
			if (!current4.isActive)
			{
				continue;
			}
			if (_behaviour.BuildingSpawnsAreAffectedByOtherBuildings() && (buildingType == TileContentType.Carpark || buildingType == TileContentType.Destination) && current4.GroupIndex == groupIndex)
			{
				Vector2Int origin = current4.Carpark.origin;
				ChangeTileWeightsAroundBuilding(origin, 5, Fix64Consts.OneHalf, current4.Carpark.footprint, "proximity to destination");
			}
			bool flag3 = false;
			if (_behaviour.UseDestinationDeadzonesFor(CityTileType.Supply))
			{
				flag3 = flag3 || buildingType == TileContentType.House;
			}
			if (_behaviour.UseDestinationDeadzonesFor(CityTileType.Demand))
			{
				flag3 = flag3 || buildingType == TileContentType.Carpark || buildingType == TileContentType.Destination;
			}
			if (flag3)
			{
				foreach (Vector2Int item3 in TileUtilities.GetBoundsWithBoundary(current4.Carpark.TopLeftWorldCoordinate, current4.Carpark.footprint).allPositionsWithin)
				{
					if (_placeableArea.Contains(item3))
					{
						Vector2Int vector2Int6 = item3 - _placeableArea.min;
						int index3 = vector2Int6.x + vector2Int6.y * _placeableArea.width;
						_placeableTileWeights[index3] = -Fix64.One;
					}
				}
			}
			if ((_buildingType == TileContentType.Destination || _buildingType == TileContentType.Carpark) && !_city.Rules.AllowConnectingDriveways)
			{
				if (current4.Carpark.entranceAtTopLeft)
				{
					ForcePositionInvalidDueToDriveway(current4.Carpark.TopLeftDrivewayTileCoordinates);
				}
				if (current4.Carpark.entranceAtBottomRight)
				{
					ForcePositionInvalidDueToDriveway(current4.Carpark.BottomRightDrivewayTileCoordinates);
				}
			}
		}
	}

	private Fix64 GetBaseWeightForTile(Vector2Int tileCoordinates, WeightEvaluationLevel weightEvaluationLevel, BuildingSpawningTileWeights tileWeights)
	{
		Fix64 value = -Fix64.One;
		if (weightEvaluationLevel == WeightEvaluationLevel.IgnoreWeights)
		{
			value = Fix64.One;
		}
		else if (tileWeights != null && !tileWeights.weights.TryGetValue((Vector3Int)tileCoordinates, out value))
		{
			switch (weightEvaluationLevel)
			{
			case WeightEvaluationLevel.ExclusivelyUseWeightedTiles:
				value = -Fix64.One;
				break;
			case WeightEvaluationLevel.AllowNonWeightedTiles:
				value = Fix64.Zero;
				break;
			default:
				Diagnostics.FailAssert("This should never logically happen!");
				break;
			}
		}
		return value;
	}

	private void ChangeTileWeightsAroundBuilding(Vector2Int buildingOrigin, int radius, Fix64 blurFactor, Vector2Int tileFootprint, string context)
	{
		for (int i = 1; i < radius; i++)
		{
			Fix64 exp = (Fix64)(radius - i);
			Fix64 fix = Fix64.Pow(blurFactor, exp);
			Vector2Int vector2Int = Vector2Int.up * i + Vector2Int.right * i + buildingOrigin + (tileFootprint - Vector2Int.one);
			Vector2Int vector2Int2 = Vector2Int.down * i + Vector2Int.left * i + buildingOrigin;
			Vector2Int vector2Int3 = new Vector2Int(vector2Int2.x, vector2Int.y);
			Vector2Int vector2Int4 = new Vector2Int(vector2Int.x, vector2Int2.y);
			int num = vector2Int.x - vector2Int2.x;
			for (int j = 0; j < num; j++)
			{
				Vector2Int position = vector2Int4 + Vector2Int.left * j;
				Vector2Int position2 = vector2Int3 + Vector2Int.right * j;
				if (_placeableArea.Contains(position))
				{
					position -= _placeableArea.min;
					int index = position.x + position.y * _placeableArea.width;
					if (_placeableTileWeights[index] >= Fix64.Zero)
					{
						_placeableTileWeights[index] *= fix;
					}
				}
				if (_placeableArea.Contains(position2))
				{
					position2 -= _placeableArea.min;
					int index2 = position2.x + position2.y * _placeableArea.width;
					if (_placeableTileWeights[index2] >= Fix64.Zero)
					{
						_placeableTileWeights[index2] *= fix;
					}
				}
			}
			int num2 = vector2Int.y - vector2Int2.y;
			for (int k = 0; k < num2; k++)
			{
				Vector2Int position3 = vector2Int2 + Vector2Int.up * k;
				Vector2Int position4 = vector2Int + Vector2Int.down * k;
				if (_placeableArea.Contains(position4))
				{
					position4 -= _placeableArea.min;
					int index3 = position4.x + position4.y * _placeableArea.width;
					if (_placeableTileWeights[index3] >= Fix64.Zero)
					{
						_placeableTileWeights[index3] *= fix;
					}
				}
				if (_placeableArea.Contains(position3))
				{
					position3 -= _placeableArea.min;
					int index4 = position3.x + position3.y * _placeableArea.width;
					if (_placeableTileWeights[index4] >= Fix64.Zero)
					{
						_placeableTileWeights[index4] *= fix;
					}
				}
			}
		}
	}

	public bool GenerateFixedPlacement(Layout layout, Vector2Int fixedCoordinates)
	{
		if ((_city.Rules.DoesIgnorePlayableArea() || TryCalculateAverageWeightOverTiles(layout, fixedCoordinates - _placeableArea.min, out var _)) && TryGeneratePlacementForLayoutAtCoordinates(layout, fixedCoordinates, Fix64.One) && (_city.Rules.DoesIgnorePlayableArea() || PlacementDrivewaysAreFree(_possiblePlacements[0])))
		{
			return true;
		}
		if (!Diagnostics.Verify(!_city.Rules.DoesIgnorePlayableArea(), "We couldn't place a fixed building in a city that ignores the playable area (i.e. menu city). Try repositioning it."))
		{
			return false;
		}
		GeneratePlacements(new List<Layout> { layout });
		_possiblePlacements.Sort((Placement a, Placement b) => Vector2Int.Distance(a.coordinates, fixedCoordinates).CompareTo(Vector2Int.Distance(b.coordinates, fixedCoordinates)));
		if (_possiblePlacements.Count > 0)
		{
			return true;
		}
		return false;
	}

	public Vector2Int GetLocalPosition(Vector2Int fixedCoordinates)
	{
		return fixedCoordinates - _placeableArea.min;
	}

	public bool GeneratePlacements(List<Layout> possibleLayouts)
	{
		bool result = false;
		_city.Rules.DoesIgnorePlayableArea();
		int num = possibleLayouts[0].footprint.x;
		int num2 = possibleLayouts[0].footprint.y;
		foreach (Layout possibleLayout in possibleLayouts)
		{
			num = Mathf.Min(num, possibleLayout.footprint.x);
			num2 = Mathf.Min(num2, possibleLayout.footprint.y);
		}
		int num3 = _placeableArea.height - Mathf.Max(0, num2 - 1);
		int num4 = _placeableArea.width - Mathf.Max(0, num - 1);
		for (int i = 0; i < num3; i++)
		{
			for (int j = 0; j < num4; j++)
			{
				if (_placeableTileWeights[i * _placeableArea.width + j] < Fix64.Zero)
				{
					continue;
				}
				Vector2Int vector2Int = Vector2Int.zero;
				bool flag = false;
				Fix64 weightFromTiles = Fix64.One;
				foreach (Layout possibleLayout2 in possibleLayouts)
				{
					bool flag2 = true;
					if (vector2Int == possibleLayout2.footprint && possibleLayout2.boatTerminalTiles.Count == 0)
					{
						flag2 = flag;
					}
					else
					{
						flag2 &= TryCalculateAverageWeightOverTiles(possibleLayout2, new Vector2Int(j, i), out weightFromTiles);
						vector2Int = possibleLayout2.footprint;
						flag = flag2;
					}
					if (flag2 && TryGeneratePlacementForLayoutAtCoordinates(possibleLayout2, new Vector2Int(_placeableArea.xMin + j, _placeableArea.yMin + i), weightFromTiles))
					{
						result = true;
					}
				}
			}
		}
		_possiblePlacements.Sort((Placement a, Placement b) => b.weight.CompareTo(a.weight));
		return result;
	}

	private bool TryCalculateAverageWeightOverTiles(Layout possibleLayout, Vector2Int localPosition, out Fix64 weightFromTiles)
	{
		weightFromTiles = Fix64.Zero;
		bool flag = true;
		int num = 0;
		for (int i = 0; i < possibleLayout.footprint.x && flag; i++)
		{
			for (int j = 0; j < possibleLayout.footprint.y; j++)
			{
				int num2 = localPosition.x + i;
				int num3 = localPosition.y + j;
				if (num2 < 0 || num2 >= _placeableArea.width)
				{
					flag = false;
					break;
				}
				if (num3 < 0 || num3 >= _placeableArea.height)
				{
					flag = false;
					break;
				}
				int num4 = num3 * _placeableArea.width + num2;
				Fix64 fix = _placeableTileWeights[num4];
				if (fix < Fix64.Zero)
				{
					flag = false;
					break;
				}
				bool flag2 = false;
				_ = Vector2Int.zero;
				foreach (Driveway driveway in possibleLayout.driveways)
				{
					Vector2Int adjacentCoordinates = TileUtilities.GetAdjacentCoordinates(new Vector2Int(localPosition.x + driveway.coordinatesOffset.x, localPosition.y + driveway.coordinatesOffset.y), driveway.direction);
					Vector2Int vector2Int = new Vector2Int(num2, num3);
					if (adjacentCoordinates == vector2Int && _placeableTileRails[num4])
					{
						flag2 = true;
						break;
					}
				}
				if (flag2)
				{
					flag = false;
					break;
				}
				if (possibleLayout.platforms.Count == 0)
				{
					if (num4 < _placeableTileRails.Count && num4 >= 0 && _placeableTileRails[num4])
					{
						flag = false;
						break;
					}
				}
				else if (_placeableTileRails[num4])
				{
					bool flag3 = false;
					foreach (RailPlatform platform in possibleLayout.platforms)
					{
						Vector2Int vector2Int2 = new Vector2Int(localPosition.x + platform.coordinatesOffset.x, localPosition.y + platform.coordinatesOffset.y);
						Vector2Int vector2Int3 = new Vector2Int(num2, num3);
						if (vector2Int2 == vector2Int3)
						{
							flag3 = true;
							break;
						}
					}
					if (!flag3)
					{
						flag = false;
						break;
					}
					num++;
				}
				weightFromTiles += fix;
			}
		}
		if (num != possibleLayout.platforms.Count)
		{
			flag = false;
		}
		weightFromTiles /= (Fix64)(possibleLayout.footprint.x * possibleLayout.footprint.y);
		return flag;
	}

	private void ForcePositionInvalidDueToDriveway(Vector2Int position)
	{
		if (_placeableArea.size.magnitude > 0f)
		{
			Vector2Int vector2Int = position - _placeableArea.min;
			int index = vector2Int.x + vector2Int.y * _placeableArea.width;
			_placeableTileDriveabilities[index] = false;
		}
	}

	private bool TryGeneratePlacementForLayoutAtCoordinates(Layout layout, Vector2Int coordinates, Fix64 weight)
	{
		bool result = false;
		bool flag = true;
		if (_placeableArea.width > 0 && _placeableArea.height > 0)
		{
			Vector2Int vector2Int = coordinates - _placeableArea.min;
			foreach (Driveway driveway in layout.driveways)
			{
				Vector2Int originCoordinates = vector2Int + driveway.coordinatesOffset;
				Vector2Int adjacentCoordinates = TileUtilities.GetAdjacentCoordinates(originCoordinates, driveway.direction);
				if (adjacentCoordinates.x < 0 || adjacentCoordinates.x >= _placeableArea.width || adjacentCoordinates.y < 0 || adjacentCoordinates.y >= _placeableArea.height)
				{
					flag = false;
					break;
				}
				int index = adjacentCoordinates.y * _placeableArea.width + adjacentCoordinates.x;
				int index2 = originCoordinates.y * _placeableArea.width + originCoordinates.x;
				Tile tile = _placeableTiles[index];
				_ = _placeableTiles[index];
				if (!_placeableTileDriveabilities[index])
				{
					flag = false;
					break;
				}
				if (_placeableTileRails[index] || _placeableTileRails[index2])
				{
					flag = false;
					break;
				}
				RoadTileNode node = new RoadTileNode(TileUtilities.GetOppositeDirection(driveway.direction), RoadType.Driveway);
				if (_city.Rules.CanBuildingsDemolishUnusedRoads && tile != null)
				{
					tile = CreateDemolishedTestTileFrom(tile);
				}
				if (tile != null && (!tile.CanSetNodeState(node, RoadState.Pending) || tile.IsNodeBlocked(node)))
				{
					flag = false;
					break;
				}
			}
		}
		if (flag)
		{
			if (_usedPlacementCount >= _placementPool.Count)
			{
				for (int i = 0; i < 20; i++)
				{
					_placementPool.Add(new Placement());
				}
			}
			Placement placement = _placementPool[_usedPlacementCount];
			_usedPlacementCount++;
			placement.coordinates = coordinates;
			placement.layout = layout;
			placement.layout.boatTerminalTiles = layout.boatTerminalTiles;
			placement.weight = _cityModel.pseudorandomGenerator.Fix64(weight);
			_possiblePlacements.Add(placement);
			result = true;
		}
		return result;
	}

	private Fix64 ScaleTileWeightByBuildingInfluence(Fix64 baseWeight, int groupIndex, Vector2Int tileCoordinates, TileContentType buildingType, int contextIndex)
	{
		int distanceToNearestDemand = _cityPlanModel.GetDistanceToNearestDemand(tileCoordinates);
		int distanceToNearestDemandOfGroup = _cityPlanModel.GetDistanceToNearestDemandOfGroup(tileCoordinates, groupIndex);
		int distanceToNearestSupplyOfGroup = _cityPlanModel.GetDistanceToNearestSupplyOfGroup(tileCoordinates, groupIndex);
		int distanceToNearestSupplyNotOfGroup = _cityPlanModel.GetDistanceToNearestSupplyNotOfGroup(tileCoordinates, groupIndex);
		if (distanceToNearestDemand <= 1 || distanceToNearestSupplyOfGroup == 0 || distanceToNearestSupplyNotOfGroup == 0)
		{
			baseWeight = Fix64.Zero;
		}
		else
		{
			switch (buildingType)
			{
			case TileContentType.Destination:
			case TileContentType.Carpark:
				if (distanceToNearestDemandOfGroup > 0 && distanceToNearestDemandOfGroup < 12)
				{
					Fix64 exp4 = (Fix64)(12 - distanceToNearestDemandOfGroup);
					if (_grouping == GroupingStyle.Far)
					{
						Fix64 fix6 = Fix64.Pow(SpawnPushFactorStrong, exp4);
						baseWeight *= fix6;
					}
					else if (_grouping == GroupingStyle.Near)
					{
						Fix64 fix7 = Fix64.Pow(SpawnPullFactor, exp4);
						baseWeight *= fix7;
					}
				}
				if (distanceToNearestSupplyOfGroup > 0 && distanceToNearestSupplyOfGroup < 7 && distanceToNearestSupplyOfGroup > 7)
				{
					Fix64 exp5 = (Fix64)(distanceToNearestSupplyOfGroup - 7);
					Fix64 fix8 = Fix64.Pow(SpawnPushFactorVeryStrong, exp5);
					baseWeight *= fix8;
				}
				break;
			case TileContentType.House:
				if (distanceToNearestSupplyOfGroup > 0 && distanceToNearestSupplyOfGroup < 7)
				{
					Fix64 exp = (Fix64)(7 - distanceToNearestSupplyOfGroup);
					if (_grouping == GroupingStyle.Far)
					{
						Fix64 fix = Fix64.Pow(SpawnPushFactorStrong, exp);
						baseWeight *= fix;
					}
					else if (_grouping == GroupingStyle.Near)
					{
						Fix64 fix2 = SpawnPullFactor;
						int nearbyHouseCountOfGroup = _cityPlanModel.GetNearbyHouseCountOfGroup(tileCoordinates, groupIndex);
						if (nearbyHouseCountOfGroup > 0)
						{
							fix2 -= (Fix64)nearbyHouseCountOfGroup * PullFactorPerNeighbourDecrease;
							fix2 = Fix64.Max(fix2, PullFactorMinimum);
						}
						Fix64 fix3 = Fix64.Pow(fix2, exp);
						baseWeight *= fix3;
					}
				}
				if (distanceToNearestDemandOfGroup > 0 && distanceToNearestDemandOfGroup < 7)
				{
					Fix64 exp2 = (Fix64)(7 - distanceToNearestDemandOfGroup);
					Fix64 fix4 = Fix64.Pow(SpawnPushFactorVeryStrong, exp2);
					baseWeight *= fix4;
				}
				if (distanceToNearestSupplyNotOfGroup > 0 && distanceToNearestSupplyNotOfGroup < 7)
				{
					Fix64 exp3 = (Fix64)((float)(7 - distanceToNearestSupplyNotOfGroup) * 0.5f);
					Fix64 fix5 = Fix64.Pow(SpawnPushFactorWeak, exp3);
					baseWeight *= fix5;
				}
				break;
			}
		}
		return baseWeight;
	}

	public Placement ChoosePlacement()
	{
		if (_possiblePlacements.Count > 0)
		{
			while (_possiblePlacements.Count > 0)
			{
				Placement placement = _possiblePlacements[0];
				if (_city.Rules.DoesIgnorePlayableArea() || PlacementDrivewaysAreFree(placement))
				{
					OnPlacementFound(0);
					return placement;
				}
				_possiblePlacements.RemoveAt(0);
			}
		}
		OnFailedPlacement();
		return null;
	}

	private bool PlacementDrivewaysAreFree(Placement placement)
	{
		List<Vector2Int> list = new List<Vector2Int>();
		foreach (Driveway driveway in placement.layout.driveways)
		{
			Vector2Int vector2Int = placement.coordinates + driveway.coordinatesOffset + TileUtilities.GetAdjacencyOffsetForDirection(driveway.direction);
			list.Clear();
			for (int i = 0; i < placement.layout.footprint.x; i++)
			{
				for (int j = 0; j < placement.layout.footprint.y; j++)
				{
					list.Add(placement.coordinates + new Vector2Int(i, j));
				}
			}
			if (!_city.Rules.AllowBlockingSpawns && !BuildingCanFindPathAwayFromDriveway(vector2Int, driveway.direction, list))
			{
				return false;
			}
			Vector2Int vector2Int2 = vector2Int - _placeableArea.min;
			int index = vector2Int2.y * _placeableArea.width + vector2Int2.x;
			if (_placeableTileRails[index])
			{
				return false;
			}
		}
		return true;
	}

	private bool BuildingCanFindPathAwayFromDriveway(Vector2Int drivewayPosition, TileDirection drivewayDirection, ICollection<Vector2Int> footprintTiles)
	{
		int num = 0;
		foreach (TileDirection radiatedDirection in TileUtilities.GetRadiatedDirections(drivewayDirection))
		{
			num++;
			Vector2Int vector2Int = TileUtilities.DirectionToTileAdjacencyOffset[(int)radiatedDirection];
			Vector2Int vector2Int2 = vector2Int * 5 + drivewayPosition;
			for (int i = 0; i < 5; i++)
			{
				Tile tile = _tilemapModel.GetTile(vector2Int2);
				if (tile == null || tile.CanDrawRoadsOn())
				{
					break;
				}
				vector2Int2 += vector2Int;
			}
			Tile tile2 = _tilemapModel.GetTile(vector2Int2);
			if (tile2 == null || tile2.CanDrawRoadsOn())
			{
				if (_pathfinder.GetPathBetweenPoints(drivewayPosition, vector2Int2, _simulation, _city, footprintTiles) != null)
				{
					return true;
				}
				if (num >= 5)
				{
					break;
				}
			}
		}
		return false;
	}

	private Tile CreateDemolishedTestTileFrom(Tile tile)
	{
		if (_testTile == null)
		{
			_testTile = _scope.Get<Tile>();
		}
		_testTile.Initialize(_tilemapModel, tile.Coordinates, tile.ContentType);
		tile.CloneInto(_testTile);
		TileModel tileModel = _tilemapModel.GetTileModel(tile.Coordinates);
		if (!Diagnostics.Verify(tileModel != null) || Passage.DoesTileHavePassage(_city.Definition, _tilemapModel, tile.Coordinates, RoadState.ActiveOrPending))
		{
			return _testTile;
		}
		TileDirectionBitfield.Enumerator enumerator = tileModel.Tile.GetTwoLaneRoads().GetEnumerator();
		while (enumerator.MoveNext())
		{
			TileDirection current = enumerator.Current;
			if (tileModel.AreAllLanesInDirectionUnused(current) && tileModel.GetAdjacentTileModelInDirection(current).Tile.ContentType != TileContentType.House)
			{
				_testTile.SetNodeState(new RoadTileNode(current), RoadState.Mothballed);
				_testTile.SetNodeState(new RoadTileNode(current), RoadState.None);
			}
		}
		return _testTile;
	}

	private bool IsTileConnectedToBuildingAndHouse(TileModel tile)
	{
		if (tile.Tile.GetTwoLaneRoadCount() == 0)
		{
			return false;
		}
		LaneModel fromLane = tile.roadChunk.lanes[0];
		if (_lanePathfinder.AreLanesConnected(fromLane, _cityPlanModel.destinationLanes, allowMothballedLaneUsage: true))
		{
			return _lanePathfinder.AreLanesConnected(fromLane, GetHouseLanes(), allowMothballedLaneUsage: true);
		}
		return false;
	}

	private IEnumerable<LaneModel> GetHouseLanes()
	{
		if (_cachedHouseLanes.Count == 0)
		{
			ModelListEnumerator<HouseModel> enumerator = _simulation.GetModels<HouseModel>().GetEnumerator();
			while (enumerator.MoveNext())
			{
				HouseModel current = enumerator.Current;
				if (current.tileModel.Tile.GetTwoLaneRoadCount() > 0)
				{
					_cachedHouseLanes.Add(current.DrivewayLane);
				}
			}
		}
		return _cachedHouseLanes;
	}

	private void OnPlacementFound(int placementIndex)
	{
	}

	private void OnFailedPlacement()
	{
	}

	[Conditional("UNITY_EDITOR")]
	private void AddTileContext(Fix64 weight, string context)
	{
		_placeableTileWeightsContext.Add($"{(float)weight:F3} {context}\n");
	}

	[Conditional("UNITY_EDITOR")]
	private void AddToTileContext(int index, Fix64 newWeight, string context)
	{
		_placeableTileWeightsContext[index] += $"{(float)newWeight:F3} {context}\n";
	}

	private string GetTileContext(int index)
	{
		return string.Empty;
	}

	public void OnReleasedFromScope(IScope scope)
	{
		if (_testTile != null)
		{
			_scope.Release(_testTile);
			_testTile = null;
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
