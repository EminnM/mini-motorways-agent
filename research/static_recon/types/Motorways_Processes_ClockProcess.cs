using Factory;
using Factory.Pools;
using FixMath;
using Motorways.Models;
using Server;

namespace Motorways.Processes;

public class ClockProcess : IProcess, IReusable
{
	public static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("ClockProcess");

	[Dependency]
	private City _city;

	public void Reset()
	{
	}

	public void Step(ISimulation simulation, Fix64 deltaTime)
	{
		ModelListEnumerator<ClockModel> enumerator = simulation.GetModels<ClockModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			ClockModel current = enumerator.Current;
			if (!current.isPaused)
			{
				current.NextFrame.time = current.CurrentFrame.time + deltaTime * _city.Rules.GetClockSpeedMultiplier();
				if (!_city.Rules.CanExpansionTimeContinue || current.expansionTimeManuallyPaused)
				{
					deltaTime = Fix64.Zero;
				}
				current.NextFrame.expansionTime = current.CurrentFrame.expansionTime + deltaTime * _city.Rules.GetClockSpeedMultiplier();
			}
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
