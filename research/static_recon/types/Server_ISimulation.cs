using Factory;
using FixMath;

namespace Server;

[Serializable(2)]
public interface ISimulation
{
	Fix64 Timestep { get; }

	bool IsPaused { get; set; }

	IScope Scope { get; }

	bool HasAnyScheduledCommands { get; }

	Command NextScheduledCommand { get; }

	bool Step();

	bool AddProcess(IProcess process);

	bool AddModel(IModel model);

	bool RemoveModel(IModel model);

	bool ContainsModel(IModel model);

	T GetModel<T>() where T : class, IModel;

	ModelList<T> GetModels<T>() where T : class, IModel;

	bool ScheduleCommand(Command command);

	void Subscribe(ISimulationObserver observer);

	bool Unsubscribe(ISimulationObserver observer);
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
