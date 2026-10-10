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

public class GenerateDemandProcess : IProcess, IReusable
{
	public enum DemandGenerationStyle
	{
		Timer,
		PermanentBalanced
	}

	[Dependency]
	private City _city;

	[Dependency]
	private CityPlanModel _cityPlanModel;

	[Dependency]
	private ClockModel _clock;

	[Dependency]
	private DemandModel _demandModel;

	[Dependency]
	private SimulationConstantsData _constants;

	[Dependency]
	private GameBehaviourModel _behaviour;

	private readonly bool[] _allocatedColorGroups = new bool[6];

	private static readonly ProfilerMarker Profiler_Step = new ProfilerMarker(ProfilerUtility.CategoryProcess, "GenerateDemandProcess.Step");

	public void Reset()
	{
	}

	public void Step(ISimulation simulation, Fix64 timestep)
	{
		BalanceDemandFromFailedSpawns(simulation);
		CalculateSpawnRamp();
		if (_city.Rules.GetDemandGenerationStyle == DemandGenerationStyle.Timer)
		{
			GenerateTimerDemand(simulation, timestep);
		}
		else
		{
			GeneratePermanentBalancedDemand(simulation);
		}
		GenerateStationDemand(simulation);
		GenerateBoatTerminalDemand(simulation);
	}

	private void BalanceDemandFromFailedSpawns(ISimulation simulation)
	{
		_demandModel.extraDemand.Clear();
		Dictionary<int, Fix64> dictionary = new Dictionary<int, Fix64>();
		Fix64 fix = DemandMultiplierForDelayedBuildings(_cityPlanModel);
		foreach (CityPlanModel.ScheduledBuilding scheduledBuilding in _cityPlanModel.scheduledBuildings)
		{
			if (scheduledBuilding.spawnAttempts > 0 && scheduledBuilding.type == CityTileType.Demand)
			{
				if (!dictionary.ContainsKey(scheduledBuilding.groupIndex))
				{
					dictionary[scheduledBuilding.groupIndex] = Fix64.Zero;
				}
				Fix64 demandMultiplier = scheduledBuilding.demandMultiplier;
				if (scheduledBuilding.spawnAttempts > _constants.MaxFailedBuildingSpawnsBeforeIgnoringWeights)
				{
					demandMultiplier *= fix;
				}
				dictionary[scheduledBuilding.groupIndex] += demandMultiplier;
			}
		}
		Fix64 zero = Fix64.Zero;
		Dictionary<int, Fix64> dictionary2 = new Dictionary<int, Fix64>();
		ModelListEnumerator<DestinationModel> enumerator2 = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator2.MoveNext())
		{
			DestinationModel current2 = enumerator2.Current;
			if (current2.isActive)
			{
				if (!dictionary2.ContainsKey(current2.GroupIndex))
				{
					dictionary2[current2.GroupIndex] = Fix64.Zero;
				}
				Fix64 demandMultiplierForBuilding = _behaviour.GetDemandMultiplierForBuilding(current2);
				dictionary2[current2.GroupIndex] += demandMultiplierForBuilding;
				zero += demandMultiplierForBuilding;
			}
		}
		foreach (KeyValuePair<int, Fix64> item in dictionary)
		{
			int key = item.Key;
			Fix64 value = item.Value;
			if (dictionary2.ContainsKey(key))
			{
				if (!_demandModel.extraDemand.ContainsKey(key))
				{
					_demandModel.extraDemand[key] = Fix64.Zero;
				}
				Fix64 fix2 = Fix64.Min(dictionary2[key], value);
				_demandModel.extraDemand[key] += fix2 / dictionary2[key];
				value -= fix2;
			}
			if (!(value > Fix64.Zero) || !(zero > Fix64.Zero))
			{
				continue;
			}
			Fix64 fix3 = value / zero;
			foreach (int key2 in dictionary2.Keys)
			{
				if (key2 != key)
				{
					if (!_demandModel.extraDemand.ContainsKey(key2))
					{
						_demandModel.extraDemand[key2] = Fix64.Zero;
					}
					_demandModel.extraDemand[key2] += fix3;
				}
			}
		}
	}

	private void CalculateSpawnRamp()
	{
		_demandModel.spawnScale = Fix64Consts.One;
		int expansionDay = _clock.ExpansionDay;
		if (_city.Definition.spawnRamp.startDay > 0)
		{
			int num = expansionDay - _city.Definition.spawnRamp.startDay;
			if (num > 0)
			{
				_demandModel.spawnScale += (Fix64)num * _city.Definition.spawnRamp.dailyIncrement * _city.Rules.SpawnRampMultiplier;
			}
		}
	}

	private void GenerateTimerDemand(ISimulation simulation, Fix64 timestep)
	{
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (!current.isActive || current.IsTrainStation || current.IsBoatTerminal)
			{
				continue;
			}
			if (current.demandTimer >= Fix64.Zero)
			{
				Fix64 fix = timestep * _demandModel.spawnScale;
				if (_demandModel.extraDemand.ContainsKey(current.GroupIndex))
				{
					fix *= Fix64.One + _demandModel.extraDemand[current.GroupIndex];
				}
				current.demandTimer -= fix;
				if (!(current.demandTimer <= Fix64.Zero))
				{
					continue;
				}
				if (CanAddDemandToDestination(current))
				{
					AddDemandToDestination(current);
				}
				else if (_city.Rules.AllowDemandRelocation)
				{
					DestinationModel destinationToReallocatePinTo = GetDestinationToReallocatePinTo(current, simulation);
					if (destinationToReallocatePinTo != null)
					{
						AddDemandToDestination(destinationToReallocatePinTo);
					}
				}
				current.demandTimer = IntervalForDestination(current);
			}
			else
			{
				current.demandTimer = IntervalForDestination(current);
			}
		}
	}

	private void GenerateStationDemand(ISimulation simulation)
	{
		ModelListEnumerator<TrainModel> enumerator = simulation.GetModels<TrainModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			TrainModel current = enumerator.Current;
			if (!current.HasPendingDemand || !Diagnostics.Verify(current.targetStation != null))
			{
				continue;
			}
			List<DestinationModel> list = new List<DestinationModel>();
			foreach (DestinationModel destination in current.targetStation.destinations)
			{
				if (Diagnostics.Verify(destination.IsTrainStation))
				{
					list.Add(destination);
				}
			}
			Fix64 zero = Fix64.Zero;
			ModelListEnumerator<HouseModel> enumerator3 = simulation.GetModels<HouseModel>().GetEnumerator();
			while (enumerator3.MoveNext())
			{
				HouseModel current3 = enumerator3.Current;
				foreach (DestinationModel item in list)
				{
					if (current3.GroupIndex == item.GroupIndex)
					{
						zero += Fix64.One;
					}
				}
			}
			Fix64 fix = zero * _constants.demandPerHouse;
			Fix64 zero2 = Fix64.Zero;
			Fix64[] array = new Fix64[list.Count];
			for (int i = 0; i < list.Count; i++)
			{
				int groupIndex = list[i].GroupIndex;
				Fix64 timeInDays = _clock.FractionalDays + _demandModel.GetGroupDemandOscillationOffset(groupIndex);
				Fix64 oscillationForDemand = _city.Definition.schedulePlanner.GetOscillationForDemand(groupIndex, timeInDays);
				zero2 += (array[i] = Fix64.One / oscillationForDemand);
			}
			for (int j = 0; j < list.Count; j++)
			{
				Fix64 value = array[j] / zero2 * fix;
				value = Fix64.Clamp(Fix64.Round(value), (Fix64)_constants.minDemandFromTrain, (Fix64)_constants.maxDemandFromTrain);
				AddDemandToStationOrTerminal(list[j], (int)(long)value);
			}
			current.HasPendingDemand = false;
		}
	}

	private void GenerateBoatTerminalDemand(ISimulation simulation)
	{
		ModelListEnumerator<BoatModel> enumerator = simulation.GetModels<BoatModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			BoatModel current = enumerator.Current;
			if (!current.HasPendingDemand || !Diagnostics.Verify(current.GetTargetTerminal() != null))
			{
				continue;
			}
			List<DestinationModel> list = new List<DestinationModel>();
			foreach (DestinationModel destination in current.GetTargetTerminal().destinations)
			{
				if (Diagnostics.Verify(destination.IsBoatTerminal))
				{
					list.Add(destination);
				}
			}
			Fix64 zero = Fix64.Zero;
			ModelListEnumerator<HouseModel> enumerator3 = simulation.GetModels<HouseModel>().GetEnumerator();
			while (enumerator3.MoveNext())
			{
				HouseModel current3 = enumerator3.Current;
				foreach (DestinationModel item in list)
				{
					if (current3.GroupIndex == item.GroupIndex)
					{
						zero += Fix64.One;
					}
				}
			}
			Fix64 fix = zero * _constants.demandPerHouse;
			Fix64 zero2 = Fix64.Zero;
			Fix64[] array = new Fix64[list.Count];
			for (int i = 0; i < list.Count; i++)
			{
				int groupIndex = list[i].GroupIndex;
				Fix64 timeInDays = _clock.FractionalDays + _demandModel.GetGroupDemandOscillationOffset(groupIndex);
				Fix64 oscillationForDemand = _city.Definition.schedulePlanner.GetOscillationForDemand(groupIndex, timeInDays);
				zero2 += (array[i] = Fix64.One / oscillationForDemand);
			}
			for (int j = 0; j < list.Count; j++)
			{
				Fix64 value = array[j] / zero2 * fix;
				value = Fix64.Clamp(Fix64.Round(value), (Fix64)_constants.minDemandFromTrain, (Fix64)_constants.maxDemandFromTrain);
				value *= _behaviour.GetDemandMultiplierForBuilding(list[j]);
				AddDemandToStationOrTerminal(list[j], (int)(long)value);
			}
			current.HasPendingDemand = false;
		}
	}

	private void GeneratePermanentBalancedDemand(ISimulation simulation)
	{
		Array.Clear(_allocatedColorGroups, 0, _allocatedColorGroups.Length);
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (!current.isActive || current.IsTrainStation || _allocatedColorGroups[current.GroupIndex])
			{
				continue;
			}
			_allocatedColorGroups[current.GroupIndex] = true;
			List<DestinationModel> list = new List<DestinationModel> { current };
			int num = current.TotalDemand;
			ModelListEnumerator<DestinationModel> enumerator2 = simulation.GetModels<DestinationModel>().GetEnumerator();
			while (enumerator2.MoveNext())
			{
				DestinationModel current2 = enumerator2.Current;
				if (!current2.IsTrainStation && !list.Contains(current2) && current.GroupIndex == current2.GroupIndex)
				{
					list.Add(current2);
					num += current2.TotalDemand;
				}
			}
			int num2 = 0;
			ModelListEnumerator<VehicleModel> enumerator3 = simulation.GetModels<VehicleModel>().GetEnumerator();
			while (enumerator3.MoveNext())
			{
				VehicleModel current3 = enumerator3.Current;
				if (current3.house.GroupIndex == current.GroupIndex && (current3.IsAvailableAtHouse || current3.IsDrivingToDestination))
				{
					num2++;
				}
			}
			int num3 = num2 - num;
			int num4 = 0;
			list.Sort((DestinationModel x, DestinationModel y) => x.TotalDemand - y.TotalDemand);
			for (int num5 = 0; num5 < num3; num5++)
			{
				if (num4 >= 30)
				{
					break;
				}
				int index = (num5 + num4) % list.Count;
				if (CanAddDemandToDestination(list[index]))
				{
					AddDemandToDestination(list[index]);
					continue;
				}
				num5--;
				num4++;
			}
		}
	}

	private DestinationModel GetDestinationToReallocatePinTo(DestinationModel maxedDestination, ISimulation simulation)
	{
		DestinationModel destinationModel = null;
		DestinationModel destinationModel2 = null;
		ModelListEnumerator<DestinationModel> enumerator = simulation.GetModels<DestinationModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			DestinationModel current = enumerator.Current;
			if (!current.isActive || !CanAddDemandToDestination(current))
			{
				continue;
			}
			if (current.GroupIndex == maxedDestination.GroupIndex)
			{
				if (destinationModel == null || Vector2.SqrMagnitude(current.Carpark.TopLeftCarparkTileCoordinate - maxedDestination.Carpark.TopLeftCarparkTileCoordinate) < Vector2.SqrMagnitude(destinationModel.Carpark.TopLeftCarparkTileCoordinate - maxedDestination.Carpark.TopLeftCarparkTileCoordinate))
				{
					destinationModel = current;
				}
			}
			else if (destinationModel2 == null || Vector2.SqrMagnitude(current.Carpark.TopLeftCarparkTileCoordinate - maxedDestination.Carpark.TopLeftCarparkTileCoordinate) < Vector2.SqrMagnitude(destinationModel2.Carpark.TopLeftCarparkTileCoordinate - maxedDestination.Carpark.TopLeftCarparkTileCoordinate))
			{
				destinationModel2 = current;
			}
		}
		return destinationModel ?? destinationModel2;
	}

	private bool CanAddDemandToDestination(DestinationModel destination)
	{
		return destination.TotalDemand < _city.Rules.GetMaximumDemandForDestination(destination);
	}

	private void AddDemandToDestination(DestinationModel destination)
	{
		if (Diagnostics.Verify(destination.GroupIndex >= 0))
		{
			destination.unassignedDemand.Add(destination.GroupIndex);
		}
	}

	private void AddDemandToStationOrTerminal(DestinationModel destination, int amountOfDemand)
	{
		if (!Diagnostics.Verify(destination.GroupIndex >= 0))
		{
			return;
		}
		int maximumDemandForDestination = _city.Rules.GetMaximumDemandForDestination(destination);
		for (int i = 0; i < amountOfDemand; i++)
		{
			if (destination.TotalDemand >= maximumDemandForDestination)
			{
				break;
			}
			destination.unassignedDemand.Add(destination.GroupIndex);
		}
	}

	private Fix64 DemandMultiplierForDelayedBuildings(CityPlanModel cityPlan)
	{
		int num = 0;
		foreach (CityPlanModel.ScheduledBuilding scheduledBuilding in cityPlan.scheduledBuildings)
		{
			if (scheduledBuilding.spawnAttempts > _constants.MaxFailedBuildingSpawnsBeforeIgnoringWeights)
			{
				num++;
			}
		}
		if (num > 0)
		{
			return Fix64.One + Fix64.Log2((Fix64)num);
		}
		return Fix64.One;
	}

	private Fix64 IntervalForDestination(DestinationModel model)
	{
		Fix64 fix = (Fix64)20.0;
		if (_clock.Time < Fix64.One)
		{
			return (Fix64)3L;
		}
		Fix64 timeInDays = _clock.FractionalDays + _demandModel.GetGroupDemandOscillationOffset(model.GroupIndex);
		Fix64 oscillationForDemand = _city.Definition.schedulePlanner.GetOscillationForDemand(model.GroupIndex, timeInDays);
		return fix / (_constants.AverageCarsPerDay * _behaviour.GetDemandMultiplierForBuilding(model) * oscillationForDemand);
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
