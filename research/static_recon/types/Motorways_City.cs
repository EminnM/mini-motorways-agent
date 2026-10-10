using System;
using System.Collections.Generic;
using Factory;
using Factory.Pools;
using FixMath;
using Motorways.Models;
using Server;
using UnityEngine;

namespace Motorways;

public class City : IReleasedFromScopeHandler, IReusable
{
	public enum PlayableAreaRoundingType
	{
		ForceWholeTiles,
		AllowPartialTiles
	}

	private CityDefinition _definition;

	private int _nextMotorwayId = 1;

	public static readonly Fix64 PlayableRatio = (Fix64)16f / (Fix64)9f;

	[Dependency]
	private ISimulation _simulation;

	[Dependency]
	private CityModel _cityModel;

	[Dependency]
	private CityPlanModel _cityPlanModel;

	[Dependency]
	private ClockModel _clockModel;

	[Dependency]
	private GameBehaviourModel _behaviour;

	[Dependency]
	public IScope Scope { get; private set; }

	public GameMode GameMode => _cityModel.Mode;

	public CityDefinition Definition
	{
		get
		{
			return _definition;
		}
		set
		{
			_definition = value;
			_definition.CompileTilemap();
		}
	}

	public GameRules Rules { get; private set; }

	public int NextMotorwayId => _nextMotorwayId;

	public string InstanceId => _cityModel.InstanceId;

	public bool Initialize(CityDefinition cityDefinition, GameRules rules)
	{
		if (!Diagnostics.Verify(_definition == null || _definition == cityDefinition, "Unable to reinitialize City with a new definition."))
		{
			return false;
		}
		Definition = cityDefinition;
		cityDefinition.CompileTilemap();
		SetGameRules(rules);
		if (Rules.ShouldGameStartFullyExpanded && _clockModel.ExpansionTime < ClockModel.DaysToSeconds(Definition.cameraZoom.durationInDays))
		{
			_clockModel.SetExpansionTimeToDay(Definition.cameraZoom.durationInDays);
		}
		_nextMotorwayId = 1;
		if (_simulation != null)
		{
			ModelListEnumerator<MotorwayModel> enumerator = _simulation.GetModels<MotorwayModel>().GetEnumerator();
			while (enumerator.MoveNext())
			{
				MotorwayModel current = enumerator.Current;
				_nextMotorwayId = Mathf.Max(_nextMotorwayId, current.Id + 1);
			}
			ModelListEnumerator<TileModel> enumerator2 = _simulation.GetModels<TileModel>().GetEnumerator();
			while (enumerator2.MoveNext())
			{
				int unbuiltMotorwayId = enumerator2.Current.Tile.UnbuiltMotorwayId;
				if (unbuiltMotorwayId != -1)
				{
					_nextMotorwayId = Mathf.Max(_nextMotorwayId, unbuiltMotorwayId + 1);
				}
			}
		}
		return true;
	}

	public void SetGameRules(GameRules newRules)
	{
		if (Rules != null && Rules != newRules)
		{
			Scope.Release(Rules);
			Rules = null;
		}
		Rules = newRules;
	}

	public void Reset()
	{
		Rules = null;
		_definition = null;
		_nextMotorwayId = 1;
	}

	public void OnReleasedFromScope(IScope scope)
	{
		if (Rules != null)
		{
			scope.Release(Rules);
			Rules = null;
		}
	}

	public int GetNextMotorwayIdAndIncrement()
	{
		_nextMotorwayId++;
		return _nextMotorwayId - 1;
	}

	public Fix64 GetCameraSizeAtTime(Fix64 time)
	{
		if (Definition == null)
		{
			return (Fix64)10L;
		}
		AnimationCurve velocity = Definition.cameraZoom.velocity;
		if (velocity == null)
		{
			return (Fix64)10L;
		}
		Fix64 fix = time / ((Fix64)(5.0 / 6.0) * (Fix64)24L);
		if (fix <= Definition.cameraZoom.delayInDays)
		{
			return Definition.cameraZoom.startSize;
		}
		if (fix >= Definition.cameraZoom.delayInDays + Definition.cameraZoom.durationInDays)
		{
			return Definition.cameraZoom.endSize;
		}
		Fix64 fix2 = (Fix64)velocity.Evaluate((float)fix);
		if (Diagnostics.Verify(fix2 > Fix64.Zero, "Camera curve is either missing or contains non-positive values."))
		{
			return fix2;
		}
		Fix64 fix3 = (fix - Definition.cameraZoom.delayInDays) / Definition.cameraZoom.durationInDays;
		return Definition.cameraZoom.startSize * (Fix64.One - fix3) + Definition.cameraZoom.endSize * fix3;
	}

	public RectFixed GetSimulationPlayableAreaAtZoom(Fix64 zoom, PlayableAreaRoundingType roundingType = PlayableAreaRoundingType.AllowPartialTiles)
	{
		Fix64 fix = PlayableRatio * zoom;
		Vector3Fixed vector3Fixed = GetPlayableAreaPositionAtZoom(zoom) / TilemapModel.TileWidth;
		if (roundingType == PlayableAreaRoundingType.AllowPartialTiles)
		{
			return new RectFixed
			{
				x = vector3Fixed.x - fix * Fix64Consts.OneHalf,
				y = vector3Fixed.y - zoom * Fix64Consts.OneHalf,
				width = fix,
				height = zoom
			};
		}
		Fix64 fix2 = Fix64.Ceiling(vector3Fixed.x - fix * Fix64Consts.OneHalf);
		Fix64 fix3 = Fix64.Ceiling(vector3Fixed.y - zoom * Fix64Consts.OneHalf);
		Fix64 fix4 = Fix64.Floor(vector3Fixed.x + fix * Fix64Consts.OneHalf);
		Fix64 fix5 = Fix64.Floor(vector3Fixed.y + zoom * Fix64Consts.OneHalf);
		return new RectFixed
		{
			x = fix2,
			y = fix3,
			width = fix4 - fix2,
			height = fix5 - fix3
		};
	}

	public RectFixed GetSimulationPlayableAreaAtTime(Fix64 time, PlayableAreaRoundingType roundingType = PlayableAreaRoundingType.AllowPartialTiles)
	{
		Fix64 cameraSizeAtTime = GetCameraSizeAtTime(time);
		return GetSimulationPlayableAreaAtZoom(cameraSizeAtTime, roundingType);
	}

	public RectFixed GetClientPlayableAreaAtZoom(Fix64 zoom, PlayableAreaRoundingType roundingType = PlayableAreaRoundingType.AllowPartialTiles)
	{
		RectFixed simulationPlayableAreaAtZoom = GetSimulationPlayableAreaAtZoom(zoom, roundingType);
		return new RectFixed
		{
			x = simulationPlayableAreaAtZoom.xMin * TilemapModel.TileWidth,
			y = simulationPlayableAreaAtZoom.yMin * TilemapModel.TileWidth,
			width = simulationPlayableAreaAtZoom.width * TilemapModel.TileWidth,
			height = simulationPlayableAreaAtZoom.height * TilemapModel.TileWidth
		};
	}

	public RectFixed GetClientPlayableAreaAtTime(Fix64 time, PlayableAreaRoundingType roundingType = PlayableAreaRoundingType.AllowPartialTiles)
	{
		Fix64 cameraSizeAtTime = GetCameraSizeAtTime(time);
		return GetClientPlayableAreaAtZoom(cameraSizeAtTime, roundingType);
	}

	public Vector3Fixed GetPlayableAreaPositionAtZoom(Fix64 zoom)
	{
		if (Definition != null)
		{
			Vector3Fixed a = Vector3Fixed.zero;
			if (_cityModel != null)
			{
				a = _cityModel.startOffset;
			}
			return Vector3Fixed.Lerp(a, Vector3Fixed.zero, GetLinearProgressOfZoom(zoom));
		}
		return Vector3Fixed.zero;
	}

	public Vector3Fixed GetPlayableAreaPositionAtTime(Fix64 time)
	{
		Fix64 cameraSizeAtTime = GetCameraSizeAtTime(time);
		return GetPlayableAreaPositionAtZoom(cameraSizeAtTime);
	}

	public Fix64 GetLinearProgressOfZoom(Fix64 zoom)
	{
		return Fix64.InverseLerp(Definition.cameraZoom.startSize, Definition.cameraZoom.endSize, zoom);
	}

	public bool IsTileInPlayableArea(Vector2Int coordinates, Fix64 time)
	{
		return GetSimulationPlayableAreaAtTime(time, PlayableAreaRoundingType.ForceWholeTiles).Contains(coordinates);
	}

	public void PopulateTrees(ISimulation simulation)
	{
		foreach (Tuple<Vector2Int, int> treeDatum in Definition.GetTreeData(_behaviour.UsesBonusTrees))
		{
			simulation.AddModel(CreateTree(treeDatum.Item2, treeDatum.Item1));
		}
	}

	public TreeModel CreateTree(int prefabIndex, Vector2Int position)
	{
		TreeModel treeModel = Scope.Get<TreeModel>();
		treeModel.Initialize(prefabIndex, Scope.Get<TilemapModel>().GetOrCreateTileModel(position));
		return treeModel;
	}

	public void SetupTrainNetwork(ISimulation simulation)
	{
		if (Definition == null)
		{
			return;
		}
		TrainNetworkDefinition trainNetworkDefinition = Definition.GetTrainNetworkDefinition();
		if (trainNetworkDefinition == null)
		{
			return;
		}
		foreach (TrainLineDefinition trainLine in trainNetworkDefinition.TrainLines)
		{
			if (!Diagnostics.Verify(trainLine.isValid || trainLine.TileCount <= 2))
			{
				continue;
			}
			TilemapModel model = _simulation.GetModel<TilemapModel>();
			TrainLineModel trainLineModel = Scope.Get<TrainLineModel>();
			trainLineModel.Initialize(trainLine.isLoop);
			for (int i = 0; i < trainLine.TileCount; i++)
			{
				Vector2Int railTileCoordinates = trainLine.GetRailTileCoordinates(i);
				TileModel orCreateTileModel = model.GetOrCreateTileModel(railTileCoordinates);
				TileDirection inputDirection = TileDirection.None;
				TileDirection outputDirection = TileDirection.None;
				if (i > 0 || trainLine.isLoop)
				{
					Vector2Int railTileCoordinates2 = trainLine.GetRailTileCoordinates((i == 0) ? (trainLine.TileCount - 1) : (i - 1));
					inputDirection = TileUtilities.GetDirectionBetweenAdjacentCoordinates(railTileCoordinates, railTileCoordinates2);
				}
				if (i < trainLine.TileCount - 1 || trainLine.isLoop)
				{
					Vector2Int railTileCoordinates3 = trainLine.GetRailTileCoordinates((i != trainLine.TileCount - 1) ? (i + 1) : 0);
					outputDirection = TileUtilities.GetDirectionBetweenAdjacentCoordinates(railTileCoordinates, railTileCoordinates3);
				}
				orCreateTileModel.Tile.SetRailConnection(new RailTileConnection(inputDirection, outputDirection));
				trainLineModel.AddTile(orCreateTileModel.RailTileModel, trainLine.GetRailTileType(i));
			}
			simulation.AddModel(trainLineModel);
		}
	}

	public void SetupBoatPathNetwork(ISimulation simulation)
	{
		if (Definition == null)
		{
			return;
		}
		BoatNetworkDefinition boatPathNetworkDefinition = Definition.GetBoatPathNetworkDefinition();
		if (boatPathNetworkDefinition == null)
		{
			return;
		}
		foreach (BoatPathLineDefinition boatLine in boatPathNetworkDefinition.BoatLines)
		{
			if (!Diagnostics.Verify(boatLine.isValid || boatLine.TileCount <= 2))
			{
				continue;
			}
			TilemapModel model = _simulation.GetModel<TilemapModel>();
			BoatPathModel boatPathModel = Scope.Get<BoatPathModel>();
			boatPathModel.Initialize(boatLine.isLoop);
			for (int i = 0; i < boatLine.TileCount; i++)
			{
				Vector2Int boatPathTileCoordinates = boatLine.GetBoatPathTileCoordinates(i);
				TileModel orCreateTileModel = model.GetOrCreateTileModel(boatPathTileCoordinates);
				TileDirection inputDirection = TileDirection.None;
				TileDirection outputDirection = TileDirection.None;
				if (i > 0 || boatLine.isLoop)
				{
					Vector2Int boatPathTileCoordinates2 = boatLine.GetBoatPathTileCoordinates((i == 0) ? (boatLine.TileCount - 1) : (i - 1));
					inputDirection = TileUtilities.GetDirectionBetweenAdjacentCoordinates(boatPathTileCoordinates, boatPathTileCoordinates2);
				}
				if (i < boatLine.TileCount - 1 || boatLine.isLoop)
				{
					Vector2Int boatPathTileCoordinates3 = boatLine.GetBoatPathTileCoordinates((i != boatLine.TileCount - 1) ? (i + 1) : 0);
					outputDirection = TileUtilities.GetDirectionBetweenAdjacentCoordinates(boatPathTileCoordinates, boatPathTileCoordinates3);
				}
				orCreateTileModel.Tile.SetBoatPathConnection(new BoatPathTileConnection(inputDirection, outputDirection));
				boatPathModel.AddTile(orCreateTileModel.BoatPathTileModel, boatLine.GetBoatPathTileType(i));
			}
			simulation.AddModel(boatPathModel);
		}
	}

	public virtual void GenerateCityLayout()
	{
		if (Definition == null)
		{
			return;
		}
		bool flag = false;
		if (Rules.HasDisabledAutomaticSpawn())
		{
			if (Definition.GetBoatPathNetworkDefinition() == null)
			{
				return;
			}
			flag = true;
		}
		PseudorandomGenerator pseudorandomGenerator = _cityModel.pseudorandomGenerator;
		foreach (ScheduleChunk scheduleChunk in Definition.schedulePlanner.scheduleChunks)
		{
			if (scheduleChunk.plannedBuildings.Count <= 0)
			{
				continue;
			}
			Fix64 fix = (Fix64)(5.0 / 6.0) * (Fix64)24L * (Fix64)scheduleChunk.startDay;
			Fix64 fix2 = (Fix64)(5.0 / 6.0) * (Fix64)24L * (Fix64)(scheduleChunk.startDay + scheduleChunk.duration);
			Fix64 fix3 = (fix2 - fix) / (Fix64)scheduleChunk.plannedBuildings.Count;
			Fix64 fix4 = fix + fix3 * Fix64Consts.OneHalf;
			Fix64[] array = new Fix64[scheduleChunk.plannedBuildings.Count];
			for (int i = 0; i < scheduleChunk.plannedBuildings.Count; i++)
			{
				array[i] = fix + pseudorandomGenerator.Fix64(fix2 - fix);
			}
			Array.Sort(array);
			List<PlannedBuilding> list = new List<PlannedBuilding>();
			list.AddRange(scheduleChunk.plannedBuildings);
			int count = scheduleChunk.plannedBuildings.Count;
			for (int j = 0; j < count && list.Count > 0; j++)
			{
				int index = j;
				if (!scheduleChunk.buildingsAreOrdered)
				{
					index = pseudorandomGenerator.Int(list.Count);
				}
				PlannedBuilding plannedBuilding = list[index];
				if (Rules.HasDisabledAutomaticSpawn())
				{
					if (!flag || plannedBuilding.carparkPreference != CarparkPreference.ForceNewBoatTerminal)
					{
						continue;
					}
					flag = false;
				}
				if (!scheduleChunk.buildingsAreOrdered)
				{
					list.RemoveAt(index);
				}
				Fix64 fix5 = (Rules.HasSpawnScheduleVariation() ? scheduleChunk.spawnVariability : Fix64Consts.Zero);
				Fix64 time = fix4 * (Fix64Consts.One - fix5) + array[j] * fix5;
				fix4 += fix3;
				if (plannedBuilding.type != CityTileType.Supply || plannedBuilding.useFixedPosition || plannedBuilding.useFixedParameters)
				{
					CityPlanModel.ScheduledBuilding scheduledBuilding = Scope.Get<CityPlanModel.ScheduledBuilding>();
					scheduledBuilding.time = time;
					scheduledBuilding.spawnAttempts = 0;
					scheduledBuilding.type = plannedBuilding.type;
					scheduledBuilding.groupIndex = plannedBuilding.groupIndex;
					scheduledBuilding.carparkPreference = plannedBuilding.carparkPreference;
					scheduledBuilding.grouping = plannedBuilding.grouping;
					scheduledBuilding.demandMultiplier = Fix64.One + (Fix64)plannedBuilding.additionalDemandMultiplier;
					scheduledBuilding.initialUpgradeLevel = 0;
					scheduledBuilding.useFixedParameters = plannedBuilding.useFixedParameters;
					scheduledBuilding.positionOverride = plannedBuilding.positionOverride;
					scheduledBuilding.entranceOverride = plannedBuilding.entranceOverride;
					scheduledBuilding.drivewayDirectionOverride = ((scheduledBuilding.type == CityTileType.Demand) ? plannedBuilding.directionOverride : plannedBuilding.drivewayDirectionOverride);
					scheduledBuilding.tutorialIdentifier = plannedBuilding.tutorialIdentifier;
					scheduledBuilding.carparkSideOverride = TileDirection.None;
					_cityPlanModel.ScheduleBuilding(scheduledBuilding);
				}
			}
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
