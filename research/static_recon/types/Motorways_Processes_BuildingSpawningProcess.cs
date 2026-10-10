using System;
using System.Collections.Generic;
using Factory;
using Factory.Pools;
using FixMath;
using Motorways.Models;
using Server;
using Unity.Profiling;
using UnityEngine;

namespace Motorways.Processes;

public class BuildingSpawningProcess : IProcess, IReusable
{
	[Dependency]
	private IScope _scope;

	[Dependency]
	private CityPlanModel _cityPlanModel;

	[Dependency]
	private TilemapModel _tilemapModel;

	[Dependency]
	private TileEditor _tileEditor;

	[Dependency]
	private ClockModel _clock;

	[Dependency]
	private City _city;

	[Dependency]
	private CityModel _cityModel;

	[Dependency]
	private BuildingPlacer _placer;

	[Dependency]
	private DemandModel _demandModel;

	[Dependency]
	private SimulationConstantsData _constants;

	[Dependency]
	private GameBehaviourModel _behaviour;

	[Dependency]
	private UpgradeDatabaseModel _upgradeDatabaseModel;

	[Dependency]
	private LaneUpdateProcess _laneUpdateProcess;

	[Dependency]
	private ReleaseMothballedLanesProcess _mothballedLanesProcess;

	public const int UpgradeAnyGroupIndex = -1;

	private static readonly Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("BuildingSpawningProcess");

	public static readonly Vector2Int HouseFootprint = new Vector2Int(1, 1);

	private static readonly List<BuildingPlacer.Layout> HouseLayouts = new List<BuildingPlacer.Layout>
	{
		new BuildingPlacer.Layout
		{
			footprint = HouseFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.North
				}
			}
		},
		new BuildingPlacer.Layout
		{
			footprint = HouseFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.East
				}
			}
		},
		new BuildingPlacer.Layout
		{
			footprint = HouseFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South
				}
			}
		},
		new BuildingPlacer.Layout
		{
			footprint = HouseFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West
				}
			}
		}
	};

	public static readonly Vector2Int DestinationFootprint = new Vector2Int(2, 2);

	private static readonly Vector2Int HorizontalCarparkFootprint = new Vector2Int(2, 3);

	private static readonly Vector2Int VerticalCarparkFootprint = new Vector2Int(3, 2);

	public static readonly Vector2Int VerticalDoubleCarparkFootprint = new Vector2Int(3, 4);

	public static readonly Vector2Int HorizontalDoubleCarparkFootprint = new Vector2Int(4, 3);

	public static readonly Vector2Int HorizontalDoubleCarparkBoatFootprint = new Vector2Int(4, 4);

	public static readonly Vector2Int VerticalDoubleCarparkBoatFootprint = new Vector2Int(4, 4);

	public static readonly List<BuildingPlacer.Layout> SingleCarparkLayouts = new List<BuildingPlacer.Layout>
	{
		new BuildingPlacer.Layout
		{
			footprint = HorizontalCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West
				}
			},
			carparkSide = TileDirection.South
		},
		new BuildingPlacer.Layout
		{
			footprint = HorizontalCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(HorizontalCarparkFootprint.x - 1, 0),
					direction = TileDirection.East
				}
			},
			carparkSide = TileDirection.South
		},
		new BuildingPlacer.Layout
		{
			footprint = VerticalCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(0, VerticalCarparkFootprint.y - 1),
					direction = TileDirection.North
				}
			},
			carparkSide = TileDirection.West
		},
		new BuildingPlacer.Layout
		{
			footprint = VerticalCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South
				}
			},
			carparkSide = TileDirection.West
		}
	};

	public static readonly List<BuildingPlacer.Layout> DoubleCarparkLayouts = new List<BuildingPlacer.Layout>
	{
		new BuildingPlacer.Layout
		{
			footprint = VerticalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.North,
					coordinatesOffset = new Vector2Int(0, VerticalDoubleCarparkFootprint.y - 1)
				}
			},
			carparkSide = TileDirection.West
		},
		new BuildingPlacer.Layout
		{
			footprint = HorizontalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.East,
					coordinatesOffset = new Vector2Int(HorizontalDoubleCarparkFootprint.x - 1, 0)
				}
			},
			carparkSide = TileDirection.South
		}
	};

	public static readonly List<BuildingPlacer.Layout> RailwayStationLayouts = new List<BuildingPlacer.Layout>
	{
		new BuildingPlacer.Layout
		{
			footprint = VerticalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.North,
					coordinatesOffset = new Vector2Int(0, VerticalDoubleCarparkFootprint.y - 1)
				}
			},
			platforms = GeneratePlatformPositions(VerticalDoubleCarparkFootprint, TileDirection.West),
			carparkSide = TileDirection.West
		},
		new BuildingPlacer.Layout
		{
			footprint = HorizontalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.East,
					coordinatesOffset = new Vector2Int(HorizontalDoubleCarparkFootprint.x - 1, 0)
				}
			},
			platforms = GeneratePlatformPositions(HorizontalDoubleCarparkFootprint, TileDirection.South),
			carparkSide = TileDirection.South
		},
		new BuildingPlacer.Layout
		{
			footprint = VerticalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South,
					coordinatesOffset = new Vector2Int(VerticalDoubleCarparkFootprint.x - 1, 0)
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.North,
					coordinatesOffset = new Vector2Int(VerticalDoubleCarparkFootprint.x - 1, VerticalDoubleCarparkFootprint.y - 1)
				}
			},
			platforms = GeneratePlatformPositions(VerticalDoubleCarparkFootprint, TileDirection.East),
			carparkSide = TileDirection.East
		},
		new BuildingPlacer.Layout
		{
			footprint = HorizontalDoubleCarparkFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West,
					coordinatesOffset = new Vector2Int(0, HorizontalDoubleCarparkFootprint.y - 1)
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.East,
					coordinatesOffset = new Vector2Int(HorizontalDoubleCarparkFootprint.x - 1, HorizontalDoubleCarparkFootprint.y - 1)
				}
			},
			platforms = GeneratePlatformPositions(HorizontalDoubleCarparkFootprint, TileDirection.North),
			carparkSide = TileDirection.North
		}
	};

	public static readonly List<BuildingPlacer.Layout> BoatTerminalLayouts = new List<BuildingPlacer.Layout>
	{
		new BuildingPlacer.Layout
		{
			footprint = HorizontalDoubleCarparkBoatFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.West
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.East,
					coordinatesOffset = new Vector2Int(HorizontalDoubleCarparkBoatFootprint.x - 1, 0)
				}
			},
			platforms = GeneratePlatformPositions(HorizontalDoubleCarparkBoatFootprint, TileDirection.South),
			boatTerminalTiles = GenerateBoatTerminalTiles(HorizontalDoubleCarparkBoatFootprint, TileDirection.South),
			carparkSide = TileDirection.South
		},
		new BuildingPlacer.Layout
		{
			footprint = VerticalDoubleCarparkBoatFootprint,
			driveways = new List<BuildingPlacer.Driveway>
			{
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.North,
					coordinatesOffset = new Vector2Int(0, VerticalDoubleCarparkBoatFootprint.y - 1)
				},
				new BuildingPlacer.Driveway
				{
					direction = TileDirection.South
				}
			},
			platforms = GeneratePlatformPositions(VerticalDoubleCarparkBoatFootprint, TileDirection.West),
			boatTerminalTiles = GenerateBoatTerminalTiles(VerticalDoubleCarparkBoatFootprint, TileDirection.West),
			carparkSide = TileDirection.West
		}
	};

	private static readonly ProfilerMarker Profiler_Step = new ProfilerMarker(ProfilerCategory.Scripts, "BuildingSpawningProcess.Step()");

	private static readonly ProfilerMarker Profiler_ScheduleHouses = new ProfilerMarker(ProfilerCategory.Scripts, "BuildingSpawningProcess.ScheduleHouses()");

	private static readonly ProfilerMarker Profiler_SpawnScheduledBuildings = new ProfilerMarker(ProfilerCategory.Scripts, "BuildingSpawningProcess.SpawnScheduledBuildings()");

	private static readonly ProfilerMarker Profiler_SpawnHouse = new ProfilerMarker(ProfilerCategory.Scripts, "BuildingSpawningProcess.SpawnHouse()");

	private static readonly ProfilerMarker Profiler_SpawnDestination = new ProfilerMarker(ProfilerCategory.Scripts, "BuildingSpawningProcess.SpawnDestination()");

	private static List<BuildingPlacer.RailPlatform> GeneratePlatformPositions(Vector2Int destinationFootprint, TileDirection carparkSide)
	{
		int num = ((carparkSide == TileDirection.North || carparkSide == TileDirection.South) ? 1 : 2);
		int num2 = ((num == 1) ? destinationFootprint.x : destinationFootprint.y);
		TileDirection direction = ((num == 1) ? TileDirection.East : TileDirection.South);
		Vector2Int vector2Int = default(Vector2Int);
		if (num == 1)
		{
			if (carparkSide == TileDirection.South)
			{
				vector2Int.y = destinationFootprint.y - 1;
			}
		}
		else
		{
			vector2Int.y = destinationFootprint.y - 1;
			if (carparkSide == TileDirection.West)
			{
				vector2Int.x = destinationFootprint.x - 1;
			}
		}
		List<BuildingPlacer.RailPlatform> list = new List<BuildingPlacer.RailPlatform>();
		TileDirection oppositeDirection = TileUtilities.GetOppositeDirection(direction);
		for (int i = 0; i < num2; i++)
		{
			Vector2Int coordinatesOffset = vector2Int + TileUtilities.GetAdjacencyOffsetForDirection(direction) * i;
			list.Add(new BuildingPlacer.RailPlatform
			{
				connection = new TileDirectionBitfield(oppositeDirection),
				coordinatesOffset = coordinatesOffset
			});
		}
		return list;
	}

	private static List<Vector2Int> GenerateBoatTerminalTiles(Vector2Int destinationFootprint, TileDirection carparkSide)
	{
		List<Vector2Int> list = new List<Vector2Int>();
		switch (carparkSide)
		{
		case TileDirection.South:
		{
			for (int j = 0; j < destinationFootprint.x; j++)
			{
				Vector2Int item2 = new Vector2Int(j, destinationFootprint.y - 1);
				list.Add(item2);
			}
			break;
		}
		case TileDirection.West:
		{
			for (int i = 0; i < destinationFootprint.y; i++)
			{
				Vector2Int item = new Vector2Int(destinationFootprint.x - 1, i);
				list.Add(item);
			}
			break;
		}
		default:
			throw new ArgumentOutOfRangeException("carparkSide", carparkSide, null);
		}
		return list;
	}

	public void Step(ISimulation simulation, Fix64 deltaTime)
	{
		if (!_city.Rules.HasDisabledAutomaticSpawn())
		{
			ScheduleHouses(simulation);
		}
		SpawnScheduledBuildings(simulation, deltaTime);
	}

	private void ScheduleHouses(ISimulation simulation)
	{
		if (_city.Rules.DoesIgnorePlayableArea() || !_cityPlanModel.IsBuildingSpawningModeSet(CityPlanModel.BuildingSpawningMode.Houses))
		{
			return;
		}
		if (_city.Rules.UsesPerCityHouseGraph)
		{
			ScheduleHousesFromCityHouseCurve(simulation);
			return;
		}
		if (_demandModel.doesSupplyNeedRecalculation)
		{
			_demandModel.RecalculateSupply();
		}
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			int groupIndex = current.GroupIndex;
			if (!_cityPlanModel.IsHouseScheduled(groupIndex) && !current.IsSupplySufficient)
			{
				Log.Info("Houses for group {0} are unable meet demand at time {1}s.", groupIndex, _clock.ExpansionTime);
				ScheduleNewHouse(groupIndex, simulation);
			}
		}
	}

	private void ScheduleHousesFromCityHouseCurve(ISimulation simulation)
	{
		int[] array = new int[MotorwaysThemeDatabase.MAX_THEME_COLOR_GROUPS];
		int[] array2 = new int[MotorwaysThemeDatabase.MAX_THEME_COLOR_GROUPS];
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (current.isActive)
			{
				array[current.GroupIndex] += current.MaximumDemandBeforeTimerStarts;
				array2[current.GroupIndex]++;
			}
		}
		bool flag = false;
		for (int i = 0; i < array2.Length; i++)
		{
			if (array2[i] > 0)
			{
				array2[i] += _city.Rules.AdditionalHousesPerGroup;
			}
			if (_cityPlanModel.IsHouseScheduled(i))
			{
				flag = true;
			}
			else if (_cityPlanModel.groupHouseCounts[i] < array2[i])
			{
				flag = true;
				ScheduleNewHouse(i, simulation);
			}
		}
		if (flag || simulation.GetModels<HouseModel>().Count >= _city.Definition.GetHousesAtDay(_clock.ExpansionDay))
		{
			return;
		}
		int num = int.MinValue;
		int groupIndex = 0;
		for (int j = 0; j < array.Length; j++)
		{
			int num2 = Mathf.CeilToInt((float)array[j] / 2f) - _cityPlanModel.groupHouseCounts[j];
			if (num2 > num && array[j] > 0)
			{
				num = num2;
				groupIndex = j;
			}
		}
		ScheduleNewHouse(groupIndex, simulation);
	}

	private void ScheduleNewHouse(int groupIndex, ISimulation simulation)
	{
		CityPlanModel.ScheduledBuilding scheduledBuilding = _scope.Get<CityPlanModel.ScheduledBuilding>();
		scheduledBuilding.type = CityTileType.Supply;
		scheduledBuilding.groupIndex = groupIndex;
		scheduledBuilding.grouping = AssessGroupingStyleForHouse(groupIndex, simulation);
		scheduledBuilding.time = _cityPlanModel.GetEarliestHouseSpawnTime(groupIndex, _clock.ExpansionTime);
		_cityPlanModel.ScheduleBuilding(scheduledBuilding);
	}

	private void SpawnScheduledBuildings(ISimulation simulation, Fix64 deltaTime)
	{
		bool flag = false;
		if (_cityPlanModel.SpawningMode != CityPlanModel.BuildingSpawningMode.All)
		{
			flag = _cityPlanModel.IsBuildingSpawningModeSet(CityPlanModel.BuildingSpawningMode.Destinations) ^ _cityPlanModel.IsBuildingSpawningModeSet(CityPlanModel.BuildingSpawningMode.Houses);
			Fix64 fix = Fix64.Zero;
			for (int i = 0; i < _cityPlanModel.scheduledBuildings.Count; i++)
			{
				CityPlanModel.ScheduledBuilding scheduledBuilding = _cityPlanModel.scheduledBuildings[i];
				if (scheduledBuilding.useFixedParameters)
				{
					flag = true;
				}
				else if (_cityPlanModel.SpawningMode == CityPlanModel.BuildingSpawningMode.None || (scheduledBuilding.type == CityTileType.Demand && !_cityPlanModel.IsBuildingSpawningModeSet(CityPlanModel.BuildingSpawningMode.Destinations)) || (scheduledBuilding.type == CityTileType.Supply && !_cityPlanModel.IsBuildingSpawningModeSet(CityPlanModel.BuildingSpawningMode.Houses)))
				{
					scheduledBuilding.time += deltaTime;
					if (scheduledBuilding.time > fix)
					{
						fix = scheduledBuilding.time;
					}
				}
			}
		}
		if (_city.Rules.FailedSpawnsIgnoreStoppedExpansionTime && !_city.Rules.CanExpansionTimeContinue)
		{
			Fix64 fix2 = Fix64.Zero;
			for (int j = 0; j < _cityPlanModel.scheduledBuildings.Count; j++)
			{
				CityPlanModel.ScheduledBuilding scheduledBuilding2 = _cityPlanModel.scheduledBuildings[j];
				if (scheduledBuilding2.useFixedParameters)
				{
					flag = true;
					continue;
				}
				if (scheduledBuilding2.spawnAttempts > 0)
				{
					scheduledBuilding2.time -= deltaTime;
					if (!flag && scheduledBuilding2.time < fix2)
					{
						flag = true;
					}
				}
				fix2 = scheduledBuilding2.time;
			}
		}
		if (flag)
		{
			_cityPlanModel.scheduledBuildings.Sort((CityPlanModel.ScheduledBuilding first, CityPlanModel.ScheduledBuilding second) => first.time.CompareTo(second.time));
		}
		Fix64 expansionTime = _clock.ExpansionTime;
		while (_cityPlanModel.scheduledBuildings.Count > 0 && _cityPlanModel.scheduledBuildings[0].time <= expansionTime)
		{
			bool flag2 = true;
			try
			{
				Log.Info("Spawning new buildings in city {0} at time {1}s. Simulation seed is {2}.", _city.Definition.name, _clock.ExpansionTime, _cityModel.pseudorandomGenerator);
				CityPlanModel.ScheduledBuilding scheduledBuilding3 = _cityPlanModel.scheduledBuildings[0];
				BuildingPlacer.WeightEvaluationLevel weightEvaluationLevel = _behaviour.GetDefaultBuildingWeightEvaluationLevel(scheduledBuilding3.type);
				if (weightEvaluationLevel != BuildingPlacer.WeightEvaluationLevel.IgnoreWeights)
				{
					weightEvaluationLevel = (scheduledBuilding3.useFixedParameters ? BuildingPlacer.WeightEvaluationLevel.IgnoreWeights : ((scheduledBuilding3.spawnAttempts > _constants.MaxFailedBuildingSpawnsBeforeIgnoringWeights) ? ((scheduledBuilding3.type != CityTileType.Demand) ? BuildingPlacer.WeightEvaluationLevel.AllowNonWeightedTiles : BuildingPlacer.WeightEvaluationLevel.AllowNonWeightedTiles) : BuildingPlacer.WeightEvaluationLevel.ExclusivelyUseWeightedTiles));
				}
				if (scheduledBuilding3.type == CityTileType.Supply)
				{
					_placer.StartPlacing(TileContentType.House, scheduledBuilding3.groupIndex, scheduledBuilding3.grouping, weightEvaluationLevel);
					if (!scheduledBuilding3.useFixedParameters)
					{
						_placer.GeneratePlacements(HouseLayouts);
					}
					else
					{
						BuildingPlacer.Driveway item = new BuildingPlacer.Driveway
						{
							direction = scheduledBuilding3.drivewayDirectionOverride
						};
						BuildingPlacer.Layout layout = new BuildingPlacer.Layout
						{
							footprint = HouseFootprint,
							driveways = new List<BuildingPlacer.Driveway> { item }
						};
						Diagnostics.Verify(_placer.GenerateFixedPlacement(layout, scheduledBuilding3.positionOverride), "Couldn't generate a valid fixed placement at {0}!", scheduledBuilding3.positionOverride);
					}
					BuildingPlacer.Placement placement = _placer.ChoosePlacement();
					if (placement != null)
					{
						Log.Info("Placed house of group {0} on tile {1} at time {2}s.", scheduledBuilding3.groupIndex, placement.coordinates, _clock.ExpansionTime);
						Tile tile = _tilemapModel.GetTile(placement.coordinates);
						if (tile != null && tile.HasTrafficLight)
						{
							TileEditResult tileEditResult = _tileEditor.ClearTileExplicit(_tilemapModel, placement.coordinates, TileEditor.ClearTileOfType.TrafficLight);
							if (tileEditResult.IsSuccessful && tileEditResult.edit != null)
							{
								Log.Info($"Clearing traffic light at {placement.coordinates} underneath house.");
								_upgradeDatabaseModel.ApplyEdit(tileEditResult.edit, _tilemapModel);
								tileEditResult.edit.ApplyToTilemap(_tilemapModel);
								tileEditResult.edit.ApplyToSimulation(simulation);
							}
						}
						TileEditResult tileEditResult2 = _tileEditor.ClearTileExplicit(_tilemapModel, placement.coordinates, TileEditor.ClearTileOfType.Roads);
						if (tileEditResult2.IsSuccessful && tileEditResult2.edit != null)
						{
							tileEditResult2.edit.ApplyToTilemap(_tilemapModel);
							tileEditResult2.edit.ApplyToSimulation(simulation);
							_upgradeDatabaseModel.ApplyEdit(tileEditResult2.edit, _tilemapModel);
							_laneUpdateProcess.Step(simulation, deltaTime);
							_mothballedLanesProcess.Step(simulation, deltaTime);
						}
						HouseModel houseModel = _scope.Get<HouseModel>();
						houseModel.Initialize(scheduledBuilding3.groupIndex, _tilemapModel.GetOrCreateTileModel(placement.coordinates), scheduledBuilding3.tutorialIdentifier);
						simulation.AddModel(houseModel);
						_cityPlanModel.RecordNewHouse(houseModel);
						_demandModel.ApplyIncrementalSupplyFromHouse(houseModel);
						Tile tile2 = houseModel.tileModel.Tile;
						TileDirection direction = placement.layout.driveways[0].direction;
						tile2.SetNodeState(new RoadTileNode(direction, RoadType.Driveway), RoadState.Pending);
						_tilemapModel.GetOrCreateTile(TileUtilities.GetAdjacentCoordinates(placement.coordinates, direction)).SetNodeState(new RoadTileNode(TileUtilities.GetOppositeDirection(direction)), RoadState.Pending);
					}
					else
					{
						Log.Info("Failed to place house of group {0} at time {1}s.", scheduledBuilding3.groupIndex, _clock.ExpansionTime);
						scheduledBuilding3.time += _constants.FailedHouseSpawnCooldown;
						scheduledBuilding3.spawnAttempts++;
						flag2 = false;
						_cityPlanModel.ScheduleBuilding(scheduledBuilding3);
					}
					continue;
				}
				if (scheduledBuilding3.grouping == GroupingStyle.Circle)
				{
					if (_behaviour.DoesBuildingStartUpgraded(scheduledBuilding3.groupIndex) || LevelUpDestinationBasedOnAge(simulation, scheduledBuilding3.groupIndex, scheduledBuilding3.demandMultiplier))
					{
						continue;
					}
					if (scheduledBuilding3.spawnAttempts > _constants.MaxFailedBuildingSpawnsBeforeIgnoringWeights)
					{
						if (scheduledBuilding3.groupIndex == -1)
						{
							scheduledBuilding3.groupIndex = ChooseRandomActiveGroupIndex(simulation);
						}
						scheduledBuilding3.grouping = GroupingStyle.Normal;
						scheduledBuilding3.spawnAttempts = 0;
						if (!_demandModel.failedDestinationUpgrades.TryGetValue(scheduledBuilding3.groupIndex, out var value))
						{
							_demandModel.failedDestinationUpgrades.Add(scheduledBuilding3.groupIndex, 1);
						}
						else
						{
							_demandModel.failedDestinationUpgrades[scheduledBuilding3.groupIndex] = value + 1;
						}
					}
					scheduledBuilding3.time += _constants.FailedDestinationRetryDelay;
					scheduledBuilding3.spawnAttempts++;
					flag2 = false;
					_cityPlanModel.ScheduleBuilding(scheduledBuilding3);
					continue;
				}
				if (scheduledBuilding3.carparkPreference == CarparkPreference.NoPreference)
				{
					if (_cityModel.pseudorandomGenerator.Fix64() <= _cityPlanModel.DoubleDestinationProbability)
					{
						scheduledBuilding3.carparkPreference = CarparkPreference.Double;
						_cityPlanModel.ResetDoubleDestinationProbability();
					}
					else
					{
						scheduledBuilding3.carparkPreference = CarparkPreference.Solo;
						_cityPlanModel.IncreaseDoubleDestinationProbability();
					}
				}
				if (scheduledBuilding3.carparkPreference != CarparkPreference.Station && scheduledBuilding3.carparkPreference != CarparkPreference.JoinStation && scheduledBuilding3.carparkPreference != CarparkPreference.ForceNewStation && scheduledBuilding3.carparkPreference != CarparkPreference.ForceNewDouble && _behaviour.ForceDoubleDestinations())
				{
					scheduledBuilding3.carparkPreference = CarparkPreference.ForceDouble;
				}
				Log.Info("Attempting to place destination with carpark preference {0}.", scheduledBuilding3.carparkPreference);
				CarparkPreference carparkPreference = scheduledBuilding3.carparkPreference;
				bool num = carparkPreference == CarparkPreference.Double || carparkPreference == CarparkPreference.ForceDouble || carparkPreference == CarparkPreference.JoinDouble || carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinStation || carparkPreference == CarparkPreference.ForceNewBoatTerminal || carparkPreference == CarparkPreference.JoinBoatTerminal;
				carparkPreference = scheduledBuilding3.carparkPreference;
				bool flag3 = carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinStation || carparkPreference == CarparkPreference.ForceNewStation;
				carparkPreference = scheduledBuilding3.carparkPreference;
				bool flag4 = carparkPreference == CarparkPreference.ForceNewBoatTerminal || carparkPreference == CarparkPreference.JoinBoatTerminal;
				if (num && HasDoubleCarparkWithVacantBuildingPosition(simulation, scheduledBuilding3, flag3, flag4, out var carparkModel) && (carparkModel.destinations[0].GroupIndex != scheduledBuilding3.groupIndex || scheduledBuilding3.useFixedParameters))
				{
					Log.Info("Placed destination of group {0} into double carpark at {1} at time {2}s.", scheduledBuilding3.groupIndex, carparkModel.TopLeftCarparkTileCoordinate, _clock.ExpansionTime);
					DestinationModel.DestinationType destinationType = (flag3 ? DestinationModel.DestinationType.TrainStation : (flag4 ? DestinationModel.DestinationType.BoatTerminal : DestinationModel.DestinationType.Destination));
					AddBuildingToDoubleCarpark(simulation, scheduledBuilding3, carparkModel, destinationType);
					continue;
				}
				BuildingPlacer.WeightSource weightSource = BuildingPlacer.WeightSource.Default;
				switch (scheduledBuilding3.carparkPreference)
				{
				case CarparkPreference.Station:
				case CarparkPreference.JoinStation:
				case CarparkPreference.ForceNewStation:
					weightSource = BuildingPlacer.WeightSource.Station;
					break;
				case CarparkPreference.ForceNewBoatTerminal:
				case CarparkPreference.JoinBoatTerminal:
					weightSource = BuildingPlacer.WeightSource.BoatTerminal;
					break;
				}
				_placer.StartPlacing(TileContentType.Carpark, scheduledBuilding3.groupIndex, scheduledBuilding3.grouping, weightEvaluationLevel, weightSource);
				if (!scheduledBuilding3.useFixedParameters)
				{
					carparkPreference = scheduledBuilding3.carparkPreference;
					if (carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinStation || carparkPreference == CarparkPreference.ForceNewStation)
					{
						_placer.GeneratePlacements(RailwayStationLayouts);
					}
					else
					{
						carparkPreference = scheduledBuilding3.carparkPreference;
						if (carparkPreference == CarparkPreference.ForceNewBoatTerminal || carparkPreference == CarparkPreference.JoinBoatTerminal)
						{
							_placer.GeneratePlacements(BoatTerminalLayouts);
						}
						else if (scheduledBuilding3.PrefersDoubleCarpark)
						{
							_placer.GeneratePlacements(DoubleCarparkLayouts);
						}
						else
						{
							_placer.GeneratePlacements(SingleCarparkLayouts);
						}
					}
				}
				else
				{
					BuildingPlacer.Layout layout2 = GenerateFixedLayoutFromPlan(scheduledBuilding3);
					_placer.GenerateFixedPlacement(layout2, scheduledBuilding3.positionOverride);
				}
				BuildingPlacer.Placement placement2 = _placer.ChoosePlacement();
				bool flag5 = false;
				if (placement2 != null)
				{
					Log.Info("Placed destination of group {0} at {1}, with footprint {2}, at time {3}s.", scheduledBuilding3.groupIndex, placement2.coordinates, placement2.layout.footprint, _clock.ExpansionTime);
					SpawnDestination(placement2, scheduledBuilding3, simulation, deltaTime);
					flag5 = true;
				}
				else
				{
					Log.Info("Failed to place destination of group {0} at time {1}s.", scheduledBuilding3.groupIndex, _clock.ExpansionTime);
				}
				if (!flag5 && scheduledBuilding3.PrefersDoubleCarpark && scheduledBuilding3.spawnAttempts > _constants.MaxFailedDoubleCarparkSpawnsBeforeConvertingToSingle)
				{
					bool flag6 = true;
					if (scheduledBuilding3.carparkPreference == CarparkPreference.ForceDouble)
					{
						for (int num2 = 1; num2 < _cityPlanModel.scheduledBuildings.Count; num2++)
						{
							CityPlanModel.ScheduledBuilding scheduledBuilding4 = _cityPlanModel.scheduledBuildings[num2];
							if (scheduledBuilding4 != scheduledBuilding3 && scheduledBuilding4.groupIndex == scheduledBuilding3.groupIndex && scheduledBuilding4.type == CityTileType.Demand)
							{
								scheduledBuilding4.carparkPreference = CarparkPreference.ForceDouble;
								flag6 = false;
								break;
							}
						}
					}
					if (flag6)
					{
						_cityPlanModel.IncreaseDoubleDestinationProbability();
					}
					scheduledBuilding3.carparkPreference = CarparkPreference.Solo;
					scheduledBuilding3.spawnAttempts = 1;
				}
				if (!flag5)
				{
					if (_city.Rules.CanUpgradeDestinationsAfterFailedSpawns && scheduledBuilding3.spawnAttempts >= _constants.MaxFailedDestinationSpawnsBeforeConvertingToUpgrade)
					{
						scheduledBuilding3.time += _constants.FailedDestinationRetryDelay;
						scheduledBuilding3.grouping = GroupingStyle.Circle;
						scheduledBuilding3.spawnAttempts = 0;
						flag2 = false;
						_cityPlanModel.ScheduleBuilding(scheduledBuilding3);
					}
					else
					{
						scheduledBuilding3.time += _constants.FailedDestinationRetryDelay;
						scheduledBuilding3.spawnAttempts++;
						flag2 = false;
						_cityPlanModel.ScheduleBuilding(scheduledBuilding3);
					}
				}
			}
			catch (Exception ex)
			{
				Diagnostics.FailAssert("{0} stacktrace: {1}", ex, ex.StackTrace);
			}
			finally
			{
				if (flag2)
				{
					_scope.Release(_cityPlanModel.scheduledBuildings[0]);
				}
				_cityPlanModel.scheduledBuildings.RemoveAt(0);
			}
		}
	}

	private void SpawnDestination(BuildingPlacer.Placement placement, CityPlanModel.ScheduledBuilding building, ISimulation simulation, Fix64 deltaTime)
	{
		EnsureMinimumTimeForDestinationSpawnsAfterBuilding(_clock.ExpansionTime + _constants.MinimumTimeBetweenDestinationSpawns, building);
		IScope scope = simulation.Scope;
		Fix64 expansionTime = _clock.ExpansionTime;
		CarparkEntrance carparkEntrance = (CarparkEntrance)0;
		foreach (BuildingPlacer.Driveway driveway in placement.layout.driveways)
		{
			carparkEntrance = ((driveway.direction != TileDirection.North && driveway.direction != TileDirection.West) ? (carparkEntrance | CarparkEntrance.BottomRight) : (carparkEntrance | CarparkEntrance.TopLeft));
		}
		bool flag = false;
		for (int i = 0; i < placement.layout.footprint.x; i++)
		{
			for (int j = 0; j < placement.layout.footprint.y; j++)
			{
				Vector2Int vector2Int = new Vector2Int(i, j) + placement.coordinates;
				Tile tile = _tilemapModel.GetTile(vector2Int);
				if (tile == null)
				{
					continue;
				}
				if (tile.HasTrafficLight)
				{
					TileEditResult tileEditResult = _tileEditor.ClearTileExplicit(_tilemapModel, vector2Int, TileEditor.ClearTileOfType.TrafficLight);
					if (tileEditResult.IsSuccessful && tileEditResult.edit != null)
					{
						flag = true;
						Log.Info($"Clearing traffic light at {vector2Int} underneath carpark.");
						_upgradeDatabaseModel.ApplyEdit(tileEditResult.edit, _tilemapModel);
						tileEditResult.edit.ApplyToTilemap(_tilemapModel);
						tileEditResult.edit.ApplyToSimulation(simulation);
					}
				}
				TileEditResult tileEditResult2 = _tileEditor.ClearTileExplicit(_tilemapModel, vector2Int, TileEditor.ClearTileOfType.Roads);
				if (tileEditResult2.IsSuccessful && tileEditResult2.edit != null)
				{
					flag = true;
					Log.Info($"Clearing Tile at {vector2Int} underneath carpark.");
					tileEditResult2.edit.ApplyToTilemap(_tilemapModel);
					tileEditResult2.edit.ApplyToSimulation(simulation);
					_upgradeDatabaseModel.ApplyEdit(tileEditResult2.edit, _tilemapModel);
				}
			}
		}
		if (flag)
		{
			_laneUpdateProcess.Step(simulation, deltaTime);
			_mothballedLanesProcess.Step(simulation, deltaTime);
		}
		CarparkModel carparkModel = scope.Get<CarparkModel>();
		TileDirection direction = placement.layout.driveways[0].direction;
		if (direction != TileDirection.East)
		{
			_ = 6;
		}
		carparkModel.Initialize(carparkEntrance, building.carparkPreference, placement);
		simulation.AddModel(carparkModel);
		_cityModel.OnCarparkAdded(carparkModel);
		Vector2Int vector2Int2 = carparkModel.destinationOffsets[0];
		DestinationModel destinationModel = scope.Get<DestinationModel>();
		CarparkPreference carparkPreference = building.carparkPreference;
		DestinationModel.DestinationType destinationType;
		if (carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinStation || carparkPreference == CarparkPreference.ForceNewStation)
		{
			destinationType = DestinationModel.DestinationType.TrainStation;
		}
		else
		{
			carparkPreference = building.carparkPreference;
			destinationType = ((carparkPreference == CarparkPreference.ForceNewBoatTerminal || carparkPreference == CarparkPreference.JoinBoatTerminal) ? DestinationModel.DestinationType.BoatTerminal : DestinationModel.DestinationType.Destination);
		}
		destinationModel.Initialize(building.groupIndex, building.demandMultiplier, DestinationFootprint, placement.coordinates + vector2Int2, carparkModel, building.tutorialIdentifier, destinationType);
		if (building.initialUpgradeLevel != 0 || (_behaviour.DoesBuildingStartUpgraded(building.groupIndex) && building.carparkPreference != CarparkPreference.Station && building.carparkPreference != CarparkPreference.JoinStation && building.carparkPreference != CarparkPreference.ForceNewStation))
		{
			destinationModel.demandLevelUpTime = expansionTime;
		}
		simulation.AddModel(destinationModel);
		_cityPlanModel.RecordNewDestination(destinationModel);
		_demandModel.ApplyAbsoluteSupplyToDestination(destinationModel);
		_demandModel.CalculateSupplyScale(destinationModel.GroupIndex);
	}

	private void EnsureMinimumTimeForDestinationSpawnsAfterBuilding(Fix64 minimumTime, CityPlanModel.ScheduledBuilding lastUnchangedBuilding)
	{
		bool flag = false;
		foreach (CityPlanModel.ScheduledBuilding scheduledBuilding in _cityPlanModel.scheduledBuildings)
		{
			if (flag)
			{
				if (!scheduledBuilding.useFixedParameters && scheduledBuilding.type == CityTileType.Demand && scheduledBuilding.time < minimumTime)
				{
					scheduledBuilding.time = minimumTime;
				}
			}
			else
			{
				flag = scheduledBuilding == lastUnchangedBuilding;
			}
		}
		_cityPlanModel.scheduledBuildings.Sort((CityPlanModel.ScheduledBuilding first, CityPlanModel.ScheduledBuilding second) => first.time.CompareTo(second.time));
	}

	private bool HasDoubleCarparkWithVacantBuildingPosition(ISimulation simulation, CityPlanModel.ScheduledBuilding requestedBuilding, bool requestStation, bool requestTerminal, out CarparkModel carparkModel)
	{
		carparkModel = null;
		ModelListEnumerator<CarparkModel> enumerator = simulation.GetModels<CarparkModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			CarparkModel current = enumerator.Current;
			if (!current.SupportsTwoDestinations || current.destinations.Count != 1)
			{
				continue;
			}
			if (requestedBuilding.useFixedParameters)
			{
				CarparkPreference carparkPreference = requestedBuilding.carparkPreference;
				if (carparkPreference == CarparkPreference.Double || carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinBoatTerminal)
				{
					if (carparkModel == null || Vector2Int.Distance(current.TopLeftWorldCoordinate, requestedBuilding.positionOverride) < Vector2Int.Distance(carparkModel.TopLeftWorldCoordinate, requestedBuilding.positionOverride))
					{
						carparkModel = current;
					}
					continue;
				}
			}
			if (current.destinations[0].IsTrainStation == requestStation && current.destinations[0].IsBoatTerminal == requestTerminal)
			{
				carparkModel = current;
				return true;
			}
		}
		return carparkModel != null;
	}

	public void AddBuildingToDoubleCarpark(ISimulation simulation, CityPlanModel.ScheduledBuilding building, CarparkModel carpark, DestinationModel.DestinationType destinationType)
	{
		IScope scope = simulation.Scope;
		Vector2Int vector2Int = carpark.destinationOffsets[1];
		DestinationModel destinationModel = scope.Get<DestinationModel>();
		destinationModel.Initialize(building.groupIndex, building.demandMultiplier, DestinationFootprint, carpark.origin + vector2Int, carpark, building.tutorialIdentifier, destinationType);
		if (_behaviour.DoesBuildingStartUpgraded(building.groupIndex) && building.carparkPreference != CarparkPreference.Station && building.carparkPreference != CarparkPreference.JoinStation && building.carparkPreference != CarparkPreference.ForceNewStation)
		{
			destinationModel.demandLevelUpTime = _clock.ExpansionTime;
		}
		else if (_behaviour.AllowSecondDestinationStartUpgraded && building.initialUpgradeLevel > 0)
		{
			destinationModel.demandLevelUpTime = _clock.ExpansionTime;
		}
		simulation.AddModel(destinationModel);
		_cityPlanModel.RecordNewDestination(destinationModel);
		_demandModel.ApplyAbsoluteSupplyToDestination(destinationModel);
		_demandModel.CalculateSupplyScale(destinationModel.GroupIndex);
	}

	private bool GetFirstSpawnTimeOfGroup(int groupIndex, ISimulation simulation, out Fix64 earliestTime)
	{
		earliestTime = Fix64.MaxValue;
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (current.GroupIndex == groupIndex && current.ActivationTime < earliestTime)
			{
				earliestTime = current.ActivationTime;
			}
		}
		return earliestTime != Fix64.MaxValue;
	}

	private GroupingStyle AssessGroupingStyleForHouse(int groupIndex, ISimulation simulation)
	{
		ScheduleGroup scheduleGroup = _city.Definition.schedulePlanner.GetScheduleGroup(groupIndex);
		Fix64 fix = ClockModel.SecondsToFractionalDays(_clock.ExpansionTime);
		if (GetFirstSpawnTimeOfGroup(groupIndex, simulation, out var earliestTime))
		{
			fix -= ClockModel.SecondsToFractionalDays(earliestTime);
		}
		float time = (float)fix;
		GroupingStyle groupingStyle = GroupingStyle.Near;
		if (_cityPlanModel.suburbCount.ContainsKey(groupIndex))
		{
			int highestNumberOfSpawnAttempts = GetHighestNumberOfSpawnAttempts(groupIndex);
			highestNumberOfSpawnAttempts = Mathf.Max(highestNumberOfSpawnAttempts - _constants.MinimumSpawnAttemptsForSuburbMultiplier, 0);
			Fix64 fix2 = Fix64.Lerp(Fix64.One, _constants.MaximumDelayedBuildingSuburbCountMultiplier, (Fix64)highestNumberOfSpawnAttempts / (Fix64)(_constants.MaximumSpawnAttemptsForSuburbMultiplier - _constants.MinimumSpawnAttemptsForSuburbMultiplier));
			Fix64 fix3 = (Fix64)_cityPlanModel.suburbCount[groupIndex];
			Fix64 fix4 = (Fix64)scheduleGroup.minimumNumSuburbs.Evaluate(time) * fix2;
			Fix64 fix5 = (Fix64)scheduleGroup.maximumNumSuburbs.Evaluate(time) * fix2;
			if (fix3 < fix4)
			{
				Fix64 fix6 = Fix64.Pow((fix4 - fix3) * _constants.MinimumSuburbCountScale, _constants.MinimumSuburbCountExponent);
				if (_cityModel.pseudorandomGenerator.Fix64() < fix6)
				{
					groupingStyle = GroupingStyle.Far;
				}
			}
			else if (fix3 < fix5)
			{
				Fix64 fix7 = Fix64.Pow((fix5 - fix3) * _constants.MaximumSuburbCountScale, _constants.MaximumSuburbCountExponent);
				if (_cityModel.pseudorandomGenerator.Fix64() < fix7)
				{
					groupingStyle = GroupingStyle.Far;
				}
			}
			if (groupingStyle == GroupingStyle.Far)
			{
				_cityPlanModel.suburbCount[groupIndex]++;
			}
		}
		else
		{
			groupingStyle = GroupingStyle.Normal;
			_cityPlanModel.suburbCount.Add(groupIndex, 1);
		}
		return groupingStyle;
	}

	private int GetHighestNumberOfSpawnAttempts(int groupIndex)
	{
		int num = 0;
		foreach (CityPlanModel.ScheduledBuilding scheduledBuilding in _cityPlanModel.scheduledBuildings)
		{
			if (scheduledBuilding.groupIndex == groupIndex && scheduledBuilding.type == CityTileType.Demand)
			{
				num = Math.Max(num, scheduledBuilding.spawnAttempts);
			}
		}
		return num;
	}

	private bool LevelUpDestinationBasedOnAge(ISimulation simulation, int groupIndex, Fix64 demandMultiplierOverride)
	{
		List<Tuple<DestinationModel, Fix64>> list = new List<Tuple<DestinationModel, Fix64>>();
		Fix64 zero = Fix64.Zero;
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if ((groupIndex != -1 && current.GroupIndex != groupIndex) || current.IsUpgraded || current.IsScheduledToBeUpgraded || current.IsTrainStation)
			{
				continue;
			}
			bool flag = false;
			for (int i = 0; i < current.Carpark.footprint.x; i++)
			{
				for (int j = 0; j < current.Carpark.footprint.y; j++)
				{
					if (_behaviour.TileSupportsCircleDestinations(groupIndex, current.Carpark.TopLeftWorldCoordinate + new Vector2Int(i, j)))
					{
						flag = true;
						break;
					}
				}
				if (flag)
				{
					break;
				}
			}
			if (flag)
			{
				Fix64 fix = _clock.ExpansionTime - current.ActivationTime;
				zero += fix;
				list.Add(Tuple.Create(current, fix));
			}
		}
		if (list.Count > 0)
		{
			Fix64 fix2 = _cityModel.pseudorandomGenerator.Fix64(zero);
			int index = list.Count - 1;
			for (int k = 0; k < list.Count; k++)
			{
				fix2 -= list[k].Item2;
				if (fix2 < Fix64.Zero)
				{
					index = k;
					break;
				}
			}
			Fix64 demandLevelUpTime = Fix64.Max(list[index].Item1.ActivationTime + CityPlanModel.MinimumTimeBeforeBuildingDemandLevelsUp, _clock.ExpansionTime);
			list[index].Item1.demandLevelUpTime = demandLevelUpTime;
			list[index].Item1.demandMultiplier = demandMultiplierOverride;
			return true;
		}
		Log.Warn("Unable to level up existing destination at group {0}, at time {1}.", groupIndex, _clock.ExpansionTime);
		return false;
	}

	public void Reset()
	{
	}

	private BuildingPlacer.Layout GenerateFixedLayoutFromPlan(CityPlanModel.ScheduledBuilding building)
	{
		BuildingPlacer.Layout layout;
		if (building.drivewayDirectionOverride == TileDirection.East)
		{
			layout = new BuildingPlacer.Layout
			{
				footprint = (building.PrefersDoubleCarpark ? HorizontalDoubleCarparkFootprint : HorizontalCarparkFootprint)
			};
			int y = ((building.carparkSideOverride == TileDirection.North) ? (layout.footprint.y - 1) : 0);
			if ((building.entranceOverride & CarparkEntrance.TopLeft) != 0)
			{
				layout.driveways.Add(new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(0, y),
					direction = TileDirection.West
				});
			}
			if ((building.entranceOverride & CarparkEntrance.BottomRight) != 0)
			{
				layout.driveways.Add(new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(layout.footprint.x - 1, y),
					direction = TileDirection.East
				});
			}
			layout.carparkSide = ((building.carparkSideOverride == TileDirection.None) ? TileDirection.South : building.carparkSideOverride);
		}
		else
		{
			layout = new BuildingPlacer.Layout
			{
				footprint = (building.PrefersDoubleCarpark ? VerticalDoubleCarparkFootprint : VerticalCarparkFootprint)
			};
			int x = 0;
			if (building.carparkSideOverride == TileDirection.East)
			{
				x = layout.footprint.x - 1;
			}
			if ((building.entranceOverride & CarparkEntrance.TopLeft) != 0)
			{
				layout.driveways.Add(new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(x, layout.footprint.y - 1),
					direction = TileDirection.North
				});
			}
			if ((building.entranceOverride & CarparkEntrance.BottomRight) != 0)
			{
				layout.driveways.Add(new BuildingPlacer.Driveway
				{
					coordinatesOffset = new Vector2Int(x, 0),
					direction = TileDirection.South
				});
			}
			layout.carparkSide = ((building.carparkSideOverride == TileDirection.None) ? TileDirection.West : building.carparkSideOverride);
		}
		CarparkPreference carparkPreference = building.carparkPreference;
		if (carparkPreference == CarparkPreference.Station || carparkPreference == CarparkPreference.JoinStation || carparkPreference == CarparkPreference.ForceNewStation)
		{
			layout.platforms = GeneratePlatformPositions(layout.footprint, layout.carparkSide);
		}
		return layout;
	}

	private int ChooseRandomActiveGroupIndex(ISimulation simulation)
	{
		List<int> list = new List<int>();
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (current.isActive && !list.Contains(current.GroupIndex))
			{
				list.Add(current.GroupIndex);
			}
		}
		if (list.Count == 0)
		{
			return 0;
		}
		return list[_cityModel.pseudorandomGenerator.Int(list.Count)];
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
