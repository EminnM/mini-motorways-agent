using Factory;

public interface IApp
{
	IScope Scope { get; }

	Game Game { get; }

	IInputState InputState { get; }

	PlayerActionController PlayerActionController { get; }

	void Start();

	void Tick(float absoluteTime, float deltaTime);

	void GameOpenedNotificationSetup();
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
