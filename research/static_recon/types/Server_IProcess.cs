using Factory;
using Factory.Pools;
using FixMath;

namespace Server;

[Serializable(1)]
public interface IProcess : IReusable
{
	void Step(ISimulation simulation, Fix64 timestep);
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
