using Client;
using Factory;
using FixMath;
using Plugins.Analytics;
using Server;

public class Game : IControllerConnectionObserver, IReleasedFromScopeHandler
{
	public static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("Game");

	private Fix64 _accumulatedTime = Fix64.Zero;

	[Dependency]
	protected ISimulation _simulation;

	[Dependency]
	protected IThemeDatabase _themeDatabase;

	protected TimeInterval _timeInterval = new TimeInterval();

	[Dependency]
	protected IClient _view;

	[Dependency]
	public IScope Scope { get; private set; }

	public GameStartReason StartReason { get; private set; }

	public ISimulation Simulation => _simulation;

	public void OnControllerConnected(IController controller)
	{
		if (CanInteract())
		{
			controller.RegisterInputActionsForGame(Scope);
			controller.EnsureActionsAreRegistered(Scope);
		}
	}

	public void OnControllerDisconnected(IController controller)
	{
		if (typeof(IScopeObserver).IsAssignableFrom(controller.GetType()))
		{
			Scope.Unsubscribe((IScopeObserver)controller);
		}
	}

	public virtual void OnReleasedFromScope(IScope scope)
	{
		_themeDatabase.RemoveView(_view);
		StopAudio();
	}

	public void Start(GameStartReason gameStartReason)
	{
		StartReason = gameStartReason;
		_themeDatabase.AddView(_view);
		_view.Start();
		_accumulatedTime = _simulation.Timestep;
	}

	public virtual void AddArbitraryAccumulatedTime(Fix64 additionalAccumulatedTime)
	{
		_accumulatedTime += additionalAccumulatedTime;
	}

	public virtual void OnGameStarted()
	{
		Scope.Get<IInputState>().SubscribeToControllerConnectionMessages(this);
	}

	public virtual void OnGameEnd(GameEndReason gameEndReason)
	{
		Scope.Get<IInputState>().UnsubscribeFromControllerConnectionMessages(this);
		Scope.Get<PlayerActionController>().GameEnded();
	}

	public virtual bool TrySave(GameJournalMotive motive)
	{
		return false;
	}

	public virtual void Tick(float frameTime)
	{
		_timeInterval.UnsyncedDelta = frameTime;
		_timeInterval.Delta = frameTime;
		AdjustTimeInterval(_timeInterval);
		_accumulatedTime += (Fix64)_timeInterval.UnpausedScaledDelta;
		while (_accumulatedTime >= _simulation.Timestep)
		{
			_simulation.Step();
			_accumulatedTime -= _simulation.Timestep;
		}
		float stepAlpha = (float)(_accumulatedTime / _simulation.Timestep);
		_view.Tick(_timeInterval, stepAlpha);
	}

	public virtual void SetPaused(bool isPaused, AnalyticsTypes.PauseReason pauseReason)
	{
		Log.Info("{0} the simulation.", isPaused ? "Pausing" : "Resuming");
		bool isPaused2 = _timeInterval.IsPaused;
		_simulation.ScheduleCommand(SetPausedCommand.Create(Scope, isPaused));
		_timeInterval.IsPaused = isPaused;
		if (isPaused2 != isPaused)
		{
			OnPauseChanged(isPaused, _timeInterval.Scale, pauseReason.ToString());
		}
	}

	public virtual void StopAudio()
	{
	}

	protected virtual void AdjustTimeInterval(TimeInterval timeInterval)
	{
	}

	public virtual bool CanInteract()
	{
		return true;
	}

	public void SetTimeScale(TimeScale scale)
	{
		TimeScale scale2 = _timeInterval.Scale;
		if (scale != scale2)
		{
			_timeInterval.Scale = scale;
			OnTimeScaleChanged(scale2, scale);
		}
	}

	public TimeScale GetTimeScale()
	{
		return _timeInterval.Scale;
	}

	protected virtual void OnTimeScaleChanged(TimeScale prevScale, TimeScale scale)
	{
	}

	protected virtual void OnPauseChanged(bool isPaused, TimeScale timeScale, string pauseReason = "")
	{
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
