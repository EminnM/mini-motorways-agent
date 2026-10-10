using Factory;

public class ProcessInputEventCommand : AppCommand, IReleasedFromScopeHandler
{
	private InputEvent _inputEvent;

	public bool Configure(float timestamp, InputEvent inputEvent)
	{
		base.Timestamp = timestamp;
		_inputEvent = inputEvent;
		return true;
	}

	public override void Reset()
	{
		_inputEvent = null;
	}

	public override bool Execute(IApp receiver)
	{
		receiver.InputState.OnInputEvent(base.Timestamp, _inputEvent);
		return true;
	}

	public void OnReleasedFromScope(IScope scope)
	{
		if (_inputEvent != null)
		{
			scope.Release(_inputEvent);
			_inputEvent = null;
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
