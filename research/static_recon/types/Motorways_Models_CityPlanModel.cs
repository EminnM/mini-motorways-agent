using System;
using System.Collections.Generic;
using Factory;
using Factory.Pools;
using FixMath;
using Motorways.Processes;
using Server;
using UnityEngine;

namespace Motorways.Models;

public class CityPlanModel : IModel, IReusable, IReleasedFromScopeHandler
{
	[Flags]
	public enum BuildingSpawningMode
	{
		None = 0,
		Houses = 1,
		Destinations = 2,
		All = 3
	}

	[Factory.Serializable(1)]
	public class ScheduledBuilding : IReusable
	{
		public Fix64 time;

		public int spawnAttempts;

		public CityTileType type;

		public int groupIndex;

		public CarparkPreference carparkPreference;

		public GroupingStyle grouping;

		public Fix64 demandMultiplier;

		public int initialUpgradeLevel;

		public bool useFixedParameters;

		public Vector2Int positionOverride;

		public CarparkEntrance entranceOverride;

		public TileDirection drivewayDirectionOverride;

		public TileDirection carparkSideOverride;

		public TutorialIdentifier tutorialIdentifier;

		public bool PrefersDoubleCarpark
		{
			get
			{
				CarparkPreference carparkPreference = this.carparkPreference;
				return carparkPreference == CarparkPreference.Double || carparkPreference == CarparkPreference.ForceDouble || carparkPreference == CarparkPreference.ForceNewDouble || carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.ForceNewStation || carparkPreference == CarparkPreference.ForceNewBoatTerminal;
			}
		}

		public void Reset()
		{
			time = Fix64Consts.Zero;
			spawnAttempts = 0;
			type = CityTileType.Demand;
			groupIndex = 0;
			carparkPreference = CarparkPreference.NoPreference;
			grouping = GroupingStyle.Normal;
			demandMultiplier = Fix64Consts.Zero;
			initialUpgradeLevel = 0;
			useFixedParameters = false;
			positionOverride = Vector2Int.zero;
			entranceOverride = CarparkEntrance.TopLeft;
			drivewayDirectionOverride = TileDirection.North;
			carparkSideOverride = TileDirection.None;
			tutorialIdentifier = TutorialIdentifier.None;
		}
	}

	public readonly List<ScheduledBuilding> scheduledBuildings = new List<ScheduledBuilding>();

	public readonly Dictionary<int, Fix64> latestHouseSpawnTime = new Dictionary<int, Fix64>();

	public readonly int[] groupHouseCounts = new int[MotorwaysThemeDatabase.MAX_THEME_COLOR_GROUPS];

	public readonly Dictionary<int, int> suburbCount = new Dictionary<int, int>();

	private readonly Dictionary<int, TileMatrixInt> _nearbyHouseCountOfGroup = new Dictionary<int, TileMatrixInt>();

	private readonly Dictionary<int, TileMatrixInt> _distanceToNearestHouseOfGroup = new Dictionary<int, TileMatrixInt>();

	private readonly Dictionary<int, TileMatrixInt> _distanceToNearestDestinationOfGroup = new Dictionary<int, TileMatrixInt>();

	public readonly List<LaneModel> destinationLanes = new List<LaneModel>();

	private const int InfiniteDistance = int.MaxValue;

	[Dependency]
	private IScope _scope;

	[Dependency]
	private City _city;

	[Dependency]
	private ClockModel _clock;

	[Dependency]
	private SimulationConstantsData _constants;

	public BuildingSpawningMode SpawningMode = BuildingSpawningMode.All;

	public static readonly Fix64 MinimumTimeBeforeBuildingDemandLevelsUp = (Fix64)10L;

	private static readonly Fix64 BaselineDoubleDestinationProbability = (Fix64)0.3;

	private static readonly Fix64 DoubleDestinationProbabilityIncrease = (Fix64)0.05;

	[Serialize(true, null)]
	public Fix64 DoubleDestinationProbability { get; private set; } = BaselineDoubleDestinationProbability;

	public bool IsBuildingSpawningModeSet(BuildingSpawningMode mode)
	{
		return (SpawningMode & mode) == mode;
	}

	public void OnReleasedFromScope(IScope scope)
	{
		foreach (ScheduledBuilding scheduledBuilding in scheduledBuildings)
		{
			scope.Release(scheduledBuilding);
		}
		scheduledBuildings.Clear();
		foreach (TileMatrixInt value in _nearbyHouseCountOfGroup.Values)
		{
			scope.Release(value);
		}
		_nearbyHouseCountOfGroup.Clear();
		foreach (TileMatrixInt value2 in _distanceToNearestHouseOfGroup.Values)
		{
			scope.Release(value2);
		}
		_distanceToNearestHouseOfGroup.Clear();
		foreach (TileMatrixInt value3 in _distanceToNearestDestinationOfGroup.Values)
		{
			scope.Release(value3);
		}
		_distanceToNearestDestinationOfGroup.Clear();
	}

	public void Reset()
	{
		scheduledBuildings.Clear();
		latestHouseSpawnTime.Clear();
		Array.Clear(groupHouseCounts, 0, groupHouseCounts.Length);
		suburbCount.Clear();
		SpawningMode = BuildingSpawningMode.All;
		DoubleDestinationProbability = BaselineDoubleDestinationProbability;
		destinationLanes.Clear();
		_nearbyHouseCountOfGroup.Clear();
		_distanceToNearestHouseOfGroup.Clear();
		_distanceToNearestDestinationOfGroup.Clear();
	}

	public bool IsHouseScheduled(int groupIndex)
	{
		foreach (ScheduledBuilding scheduledBuilding in scheduledBuildings)
		{
			if (scheduledBuilding.type == CityTileType.Supply && scheduledBuilding.groupIndex == groupIndex)
			{
				return true;
			}
		}
		return false;
	}

	public Fix64 GetEarliestHouseSpawnTime(int groupIndex, Fix64 earliestTime)
	{
		if (latestHouseSpawnTime.TryGetValue(groupIndex, out var value))
		{
			return Fix64.Max(earliestTime, value + _constants.DelayBetweenSameGroupHouseSpawns);
		}
		return earliestTime;
	}

	public int GetNearbyHouseCountOfGroup(Vector2Int tileCoordinates, int groupIndex)
	{
		if (_nearbyHouseCountOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value[tileCoordinates];
		}
		return 0;
	}

	public TileMatrixInt GetHouseDistanceMatrixForGroup(int groupIndex)
	{
		if (_distanceToNearestHouseOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value;
		}
		return null;
	}

	public TileMatrixInt GetHouseCountMatrixForGroup(int groupIndex)
	{
		if (_nearbyHouseCountOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value;
		}
		return null;
	}

	public TileMatrixInt GetDestinationMatrixForGroup(int groupIndex)
	{
		if (_distanceToNearestDestinationOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value;
		}
		return null;
	}

	public int GetDistanceToNearestSupplyOfGroup(Vector2Int tileCoordinates, int groupIndex)
	{
		if (_distanceToNearestHouseOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value[tileCoordinates];
		}
		return int.MaxValue;
	}

	public int GetDistanceToNearestSupplyNotOfGroup(Vector2Int tileCoordinates, int groupIndex)
	{
		int num = int.MaxValue;
		foreach (KeyValuePair<int, TileMatrixInt> item in _distanceToNearestHouseOfGroup)
		{
			if (item.Key != groupIndex)
			{
				int num2 = item.Value[tileCoordinates];
				if (num > num2)
				{
					num = num2;
				}
			}
		}
		return num;
	}

	public int GetDistanceToNearestDemand(Vector2Int tileCoordinates)
	{
		int num = int.MaxValue;
		foreach (TileMatrixInt value in _distanceToNearestDestinationOfGroup.Values)
		{
			int num2 = value[tileCoordinates];
			if (num > num2)
			{
				num = num2;
			}
		}
		return num;
	}

	public int GetDistanceToNearestDemandOfGroup(Vector2Int tileCoordinates, int groupIndex)
	{
		if (_distanceToNearestDestinationOfGroup.TryGetValue(groupIndex, out var value))
		{
			return value[tileCoordinates];
		}
		return int.MaxValue;
	}

	public void RecordNewHouse(HouseModel model)
	{
		List<Vector2Int> startCoordinates = new List<Vector2Int> { model.tileModel.Coordinates };
		int groupIndex = model.GroupIndex;
		if (!_distanceToNearestHouseOfGroup.TryGetValue(groupIndex, out var value))
		{
			value = TileMatrixInt.Create(_scope, _city.Definition.PlayableArea, int.MaxValue);
			_distanceToNearestHouseOfGroup.Add(groupIndex, value);
		}
		value.FloodFill(startCoordinates, 0, (int color) => color + 1, CanFloodFillEnterCell);
		if (!_nearbyHouseCountOfGroup.TryGetValue(groupIndex, out var value2))
		{
			value2 = TileMatrixInt.Create(_scope, _city.Definition.PlayableArea, 0);
			_nearbyHouseCountOfGroup.Add(groupIndex, value2);
		}
		Vector2Int coordinates = model.tileModel.Coordinates;
		int num = Mathf.Max(coordinates.x - 5, value2.Dimensions.xMin);
		int num2 = Mathf.Max(coordinates.y - 5, value2.Dimensions.yMin);
		int num3 = Mathf.Min(coordinates.x + 5, value2.Dimensions.xMax - 1);
		int num4 = Mathf.Min(coordinates.y + 5, value2.Dimensions.yMax - 1);
		for (int num5 = num2; num5 <= num4; num5++)
		{
			for (int num6 = num; num6 <= num3; num6++)
			{
				Vector2Int vector2Int = new Vector2Int(num6, num5);
				if ((vector2Int - coordinates).sqrMagnitude <= 25)
				{
					value2[vector2Int]++;
				}
			}
		}
		groupHouseCounts[model.GroupIndex]++;
		UpdateLatestHouseSpawnTime(model.GroupIndex, _clock.ExpansionTime);
	}

	public void RecordNewDestination(DestinationModel model)
	{
		List<Vector2Int> list = new List<Vector2Int>();
		foreach (TileModel tileModel in model.TileModels)
		{
			list.Add(tileModel.Coordinates);
		}
		int groupIndex = model.GroupIndex;
		if (!_distanceToNearestDestinationOfGroup.TryGetValue(groupIndex, out var value))
		{
			value = TileMatrixInt.Create(_scope, _city.Definition.PlayableArea, int.MaxValue);
			_distanceToNearestDestinationOfGroup.Add(groupIndex, value);
		}
		value.FloodFill(list, 0, (int color) => color + 1, CanFloodFillEnterCell);
	}

	public void ResetDoubleDestinationProbability()
	{
		DoubleDestinationProbability = BaselineDoubleDestinationProbability;
	}

	public void IncreaseDoubleDestinationProbability()
	{
		DoubleDestinationProbability += DoubleDestinationProbabilityIncrease;
	}

	public void ScheduleBuilding(ScheduledBuilding building)
	{
		int num = scheduledBuildings.Count;
		while (num > 0 && scheduledBuildings[num - 1].time > building.time)
		{
			num--;
		}
		scheduledBuildings.Insert(num, building);
		if (building.type == CityTileType.Supply)
		{
			UpdateLatestHouseSpawnTime(building.groupIndex, building.time);
		}
	}

	public void ClearScheduledBuildings(IScope scope)
	{
		foreach (ScheduledBuilding scheduledBuilding in scheduledBuildings)
		{
			_scope.Release(scheduledBuilding);
		}
		scheduledBuildings.Clear();
	}

	private void UpdateLatestHouseSpawnTime(int groupIndex, Fix64 newSpawnTime)
	{
		if (latestHouseSpawnTime.TryGetValue(groupIndex, out var value))
		{
			if (newSpawnTime > value)
			{
				latestHouseSpawnTime[groupIndex] = newSpawnTime;
			}
		}
		else
		{
			latestHouseSpawnTime[groupIndex] = newSpawnTime;
		}
	}

	private bool CanFloodFillEnterCell(Vector2Int coordinate, int stepCount, int targetDistance, int replacementDistance)
	{
		if (replacementDistance < targetDistance && !_city.Definition.TileIsOverWater(coordinate) && !_city.Definition.TileIsUnderAMountain(coordinate))
		{
			return !_city.Definition.TileIsOverRail(coordinate);
		}
		return false;
	}

	private int ReplaceFloodFillColor(int data, int color)
	{
		return color;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
