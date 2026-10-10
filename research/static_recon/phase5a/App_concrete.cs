using System;
using System.Collections.Generic;
using System.Globalization;
using Client;
using Factory;
using Helpers.GameCenter;
using JetBrains.Annotations;
using Motorways;
using NotificationService.Events;
using Notifications;
using UnityEngine;

public class App : IApp, IControllerConnectionObserver
{
	[Dependency]
	private IHardwareCapabilities _hardwareCapabilities;

	[Dependency]
	private ISoftwareCapabilities _softwareCapabilities;

	[Dependency]
	private IAchievementHandler _achievementHandler;

	[Dependency]
	private IGameCenterAuthentication _gameCenterAuthentication;

	[Dependency]
	private ISystemNotificationService _systemNotificationService;

	[Dependency]
	private NotificationScheduler _notificationScheduler;

	[Dependency]
	private INotificationEventSystem _notificationEventSystem;

	[Dependency]
	private IAudioSystem _audioSystem;

	[Dependency]
	private IInputState _inputState;

	[Dependency]
	private PlayerActionController _playerActionController;

	[Dependency]
	private LocaleDatabase _localeDatabase;

	[Dependency]
	private AchievementDatabase _achievementDatabase;

	[Dependency]
	private ScreenStack _screenStack;

	[Dependency]
	private IThemeDatabase _themeDatabase;

	[Dependency]
	private ActivePlayer _activePlayer;

	[Dependency]
	private Diagnostics.StorageAuditTrail _storageAuditTrail;

	[Dependency]
	private TickRegistry _tickRegistry;

	private static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("App");

	private volatile bool _preventCodeStripping = true;

	[UsedImplicitly]
	private volatile List<Calendar> _calendars;

	[Dependency]
	public IScope Scope { get; private set; }

	public Game Game => _screenStack.GetGameIfInGame();

	public IInputState InputState => _inputState;

	public PlayerActionController PlayerActionController => _playerActionController;

	public void Start()
	{
		PreventCodeStripping();
		_hardwareCapabilities.OnAppStart();
		_softwareCapabilities.OnAppStart();
		_achievementHandler.OnAppStart();
		_gameCenterAuthentication.Authenticate();
		_systemNotificationService.Setup();
		if (FeatureToggle.IsFeatureEnabled(Feature.SoakTest))
		{
			_hardwareCapabilities.IsPreventingSleep = true;
		}
		_storageAuditTrail.IsRecordingEvents = FeatureToggle.IsFeatureEnabled(Feature.RecordStorageAuditTrail);
		_inputState.SubscribeToControllerConnectionMessages(this);
		_inputState.Start();
		Diagnostics.Verify(_localeDatabase.Load(), "Failed to load the locale database!");
		Diagnostics.Verify(_achievementDatabase.Load(), "Failed to load the achievement database!");
		bool isAudioRunning = AudioSettings.dspTime > 0.0;
		_audioSystem.Start(isAudioRunning);
		_themeDatabase.Start();
		_screenStack.Start();
		_activePlayer.PlayerChanged += _notificationScheduler.OnPlayerChanged;
		_activePlayer.DataChanged += _notificationScheduler.OnPlayerDataChanged;
		_activePlayer.PlayerChanged += OnPlayerChanged;
	}

	public void GameOpenedNotificationSetup()
	{
		_notificationEventSystem.RecordEvent(new OpenedMiniMotorways());
		_systemNotificationService.RemoveAllDeliveredNotifications();
		_systemNotificationService.ApplicationBadge = 0;
	}

	private void OnPlayerChanged(Player oldPlayer, Player newPlayer)
	{
		GameOpenedNotificationSetup();
	}

	public void Tick(float absoluteAppTime, float deltaTime)
	{
		_tickRegistry.Tick(deltaTime);
		try
		{
			_audioSystem.Tick();
		}
		catch (Exception ex)
		{
			Log.Error("Caught Exception when calling _audioSystem.Tick()\n{0}", ex);
		}
		try
		{
			_inputState.Tick(absoluteAppTime);
		}
		catch (Exception ex2)
		{
			Log.Error("Caught Exception when calling _inputState.Tick()\n{0}", ex2);
		}
		try
		{
			_playerActionController.Tick(deltaTime);
		}
		catch (Exception ex3)
		{
			Log.Error("Caught Exception when calling _playerActionController.Tick()\n{0}", ex3);
		}
		try
		{
			_screenStack.Tick(deltaTime);
		}
		catch (Exception ex4)
		{
			Log.Error("Caught Exception when calling _screenStack.Tick()\n{0}", ex4);
		}
		try
		{
			_themeDatabase.Tick(deltaTime);
		}
		catch (Exception ex5)
		{
			Log.Error("Caught Exception when calling _themeDatabase.Tick()\n{0}", ex5);
		}
	}

	public void OnControllerConnected(IController controller)
	{
		controller.RegisterInputActionsForApp(Scope);
		controller.EnsureActionsAreRegistered(Scope);
	}

	public void OnControllerDisconnected(IController controller)
	{
		if (typeof(IScopeObserver).IsAssignableFrom(controller.GetType()))
		{
			Scope.Unsubscribe((IScopeObserver)controller);
		}
	}

	private void PreventCodeStripping()
	{
		if (_preventCodeStripping)
		{
			_calendars = new List<Calendar>();
			_calendars.Add(new ChineseLunisolarCalendar());
			_calendars.Add(new JapaneseLunisolarCalendar());
			_calendars.Add(new KoreanLunisolarCalendar());
			_calendars.Add(new TaiwanLunisolarCalendar());
			_calendars.Add(new GregorianCalendar());
			_calendars.Add(new HebrewCalendar());
			_calendars.Add(new HijriCalendar());
			_calendars.Add(new JapaneseCalendar());
			_calendars.Add(new JulianCalendar());
			_calendars.Add(new KoreanCalendar());
			_calendars.Add(new PersianCalendar());
			_calendars.Add(new TaiwanCalendar());
			_calendars.Add(new ThaiBuddhistCalendar());
			_calendars.Add(new UmAlQuraCalendar());
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
