using DevTools.OnScreenDebugTools;
using Factory;
using Factory.Allocators;
using Factory.Pools;
using Localization;
using Motorways;
using Motorways.Audio;
using Motorways.UI;
using Motorways.Views;
using Popups;
using UnityEngine;

public abstract class AppContainer
{
	protected static readonly Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("AppContainer");

	private IAppCommandSource _commandSource;

	public Assembler AppAssembler { get; private set; }

	public Assembler GameAssembler { get; private set; }

	public IScope AppScope { get; private set; }

	public IApp App { get; private set; }

	public static IEnvironment Environment { get; private set; }

	public AppCommandJournal CommandJournal { get; private set; }

	public void SetEnvironment(IEnvironment environment)
	{
		Environment = environment;
		FeatureToggle.AddSource(new BuildTimeConfigSettingSource(environment));
	}

	public void CreateAssemblers()
	{
		if (Environment == null)
		{
			SetEnvironment(CreateDefaultEnvironment());
		}
		RegisterSerializers();
		AppAssembler = CreateAppAssembler();
		GameAssembler = CreateGameAssembler(AppAssembler);
		Debug.LogFormat("Assembler serializer hash codes: {0}, {1}", AppAssembler.GlobalTypeSerializerHashCode, GameAssembler.GlobalTypeSerializerHashCode);
	}

	public IScope CreateScope()
	{
		AppScope = new Scope(AppAssembler);
		RegisterStorableTypeHandlers();
		return AppScope;
	}

	public IApp CreateApp()
	{
		App = AppScope.Get<IApp>();
		return App;
	}

	public void Start(bool recordJournal = false)
	{
		if (recordJournal)
		{
			CommandJournal = AppScope.Get<AppCommandJournal>();
		}
		_commandSource = AppScope.Get<IAppCommandSource>();
		_commandSource.Start();
		App.Start();
	}

	public void Tick()
	{
		foreach (IAppCommand frameCommand in _commandSource.GetFrameCommands())
		{
			CommandJournal?.Record(frameCommand);
			frameCommand.Execute(App);
			if (CommandJournal == null)
			{
				AppScope.Release(frameCommand);
			}
		}
	}

	protected virtual void RegisterSerializers()
	{
	}

	protected virtual void RegisterStorableTypeHandlers()
	{
		IStorableTypeHandlerRegistry storableTypeHandlerRegistry = AppScope.Get<IStorableTypeHandlerRegistry>();
		storableTypeHandlerRegistry.RegisterHandler<ILegacyUserProfile>(AppScope.Get<UserProfileStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<IExtendedUserProfile>(AppScope.Get<ExtendedUserProfileStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<IDeviceSettings>(AppScope.Get<DeviceSettingsStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<IGameJournalSave>(AppScope.Get<SavedGameStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<ReadonlyLegacyUserProfile>(AppScope.Get<ReadonlyUserProfileStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<ReadonlyExtendedUserProfile>(AppScope.Get<ReadonlyExtendedUserProfileStorableTypeHandler>());
		storableTypeHandlerRegistry.RegisterHandler<ReadonlyDeviceSettings>(AppScope.Get<ReadonlyDeviceSettingsStorableTypeHandler>());
	}

	protected virtual Assembler CreateAppAssembler()
	{
		Assembler assembler = new Assembler("app");
		assembler.IsValidatingObjectScrubbing = Application.isEditor;
		assembler.Register<LocaleDatabase>().Allocator(new HeapAllocator<LocaleDatabase>()).Binding(Binding.Scope);
		assembler.Register<Diagnostics.StorageAuditTrail>().Allocator(new HeapAllocator<Diagnostics.StorageAuditTrail>()).Binding(Binding.Scope);
		assembler.Register<IStorableTypeHandlerRegistry, StorableTypeHandlerRegistry>().Allocator(new HeapAllocator<StorableTypeHandlerRegistry>()).Binding(Binding.Scope);
		assembler.Register<UserProfileStorableTypeHandler>().Allocator(new HeapAllocator<UserProfileStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<ExtendedUserProfileStorableTypeHandler>().Allocator(new HeapAllocator<ExtendedUserProfileStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<ReadonlyExtendedUserProfileStorableTypeHandler>().Allocator(new HeapAllocator<ReadonlyExtendedUserProfileStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<ReadonlyUserProfileStorableTypeHandler>().Allocator(new HeapAllocator<ReadonlyUserProfileStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<ReadonlyDeviceSettingsStorableTypeHandler>().Allocator(new HeapAllocator<ReadonlyDeviceSettingsStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<DeviceSettingsStorableTypeHandler>().Allocator(new HeapAllocator<DeviceSettingsStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<SavedGameStorableTypeHandler>().Allocator(new HeapAllocator<SavedGameStorableTypeHandler>()).Binding(Binding.Scope);
		assembler.Register<IPersistentStorageService, PersistentStorageService>().Allocator(new HeapAllocator<PersistentStorageService>()).Binding(Binding.Scope);
		assembler.Register<IOAuthClient, BrowserOAuthClient>().Allocator(new HeapAllocator<BrowserOAuthClient>()).Binding(Binding.Scope);
		assembler.Register<ISteamCloudSyncService, SteamworksCloudSyncService>().Allocator(new HeapAllocator<SteamworksCloudSyncService>()).Binding(Binding.Scope);
		assembler.Register<PlayerDatabase>().Allocator(new HeapAllocator<PlayerDatabase>()).Binding(Binding.Scope);
		assembler.Register<Player>().Allocator(new HeapAllocator<Player>());
		assembler.Register<IActivePlayer, ActivePlayer>().Allocator(new HeapAllocator<ActivePlayer>()).Binding(Binding.Scope);
		assembler.Register<StringKey, MotorwaysStringKey>().Allocator(new StringKeyPool<MotorwaysStringKey>
		{
			InitialSize = 10000,
			BlockSize = 1000
		});
		assembler.Register<StandaloneLocString>().Allocator(new StringPool<StandaloneLocString>
		{
			InitialSize = 10000,
			BlockSize = 1000
		});
		assembler.Register<IApp, App>().Allocator(new HeapAllocator<App>()).Binding(Binding.Scope);
		assembler.Register<TickRegistry>().Allocator(new HeapAllocator<TickRegistry>()).Binding(Binding.Scope);
		assembler.Register<IAudioSystem, AudioSystem>().Allocator(new HeapAllocator<AudioSystem>()).Binding(Binding.Scope);
		assembler.Register<HapticFeedbackGenerator>().Allocator(new HeapAllocator<HapticFeedbackGenerator>()).Binding(Binding.Scope);
		assembler.Register<IInputState, InputState>().Allocator(new HeapAllocator<InputState>()).Binding(Binding.Scope);
		assembler.Register<IPointerState, PointerState>().Allocator(new HeapAllocator<PointerState>());
		assembler.Register<IMouseController, MouseController>().Allocator(new HeapAllocator<MouseController>()).Binding(Binding.Scope);
		assembler.Register<IKeyboardController, KeyboardController>().Allocator(new HeapAllocator<KeyboardController>()).Binding(Binding.Scope);
		assembler.Register<ITouchScreenController, TouchScreenController>().Allocator(new HeapAllocator<TouchScreenController>()).Binding(Binding.Scope);
		assembler.Register<IAppleTVRemoteController, AppleTVRemoteController>().Allocator(new HeapAllocator<AppleTVRemoteController>()).Binding(Binding.Scope);
		assembler.Register<IGamepadController, GenericGamepadController>().Allocator(new HeapAllocator<GenericGamepadController>()).Binding(Binding.Scope);
		assembler.Register<ButtonState>().Allocator(new HeapAllocator<ButtonState>());
		assembler.Register<IAppCommandSource, RuntimeAppCommandSource>().Allocator(new HeapAllocator<RuntimeAppCommandSource>()).Binding(Binding.Scope);
		assembler.Register<AppCommandJournal>().Allocator(new HeapAllocator<AppCommandJournal>()).Binding(Binding.Scope);
		assembler.Register<TickAppCommand>().Allocator(new ObjectPool<TickAppCommand>
		{
			InitialSize = 1,
			BlockSize = 60
		});
		assembler.Register<ProcessInputEventCommand>().Allocator(new ObjectPool<ProcessInputEventCommand>
		{
			InitialSize = 10,
			BlockSize = 100
		});
		assembler.Register<ConfigureDeviceCommand>().Allocator(new ObjectPool<ConfigureDeviceCommand>
		{
			InitialSize = 1
		});
		assembler.Register<InitRandomCommand>().Allocator(new ObjectPool<InitRandomCommand>
		{
			InitialSize = 1
		});
		assembler.Register<ChangeWindowFocusCommand>().Allocator(new ObjectPool<ChangeWindowFocusCommand>
		{
			InitialSize = 1
		});
		assembler.Register<PlayerActionController>().Allocator(new HeapAllocator<PlayerActionController>()).Binding(Binding.Scope);
		assembler.Register<PlayerActionGroup>().Allocator(new ObjectPool<PlayerActionGroup>
		{
			InitialSize = 10
		});
		assembler.Register<ScreenStack>().Allocator(new HeapAllocator<ScreenStack>()).Binding(Binding.Scope);
		assembler.Register<InGameMessageUIManager>().Allocator(new HeapAllocator<InGameMessageUIManager>()).Binding(Binding.Scope);
		assembler.Register<InGameMessageService>().Allocator(new HeapAllocator<InGameMessageService>()).Binding(Binding.Scope);
		assembler.Register<PopupParent>().Allocator(new GameObjectPool<PopupParent>("core", "PopupParent")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<PopupStack>().Allocator(new HeapAllocator<PopupStack>()).Binding(Binding.Scope);
		assembler.Register<NewContentIndicator>().Allocator(new GameObjectPool<NewContentIndicator>("core", "NewContentIndicator")
		{
			InitialSize = 5,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 5
		});
		GameCamera gameCamera = Object.FindObjectOfType<GameCamera>();
		if (Diagnostics.Verify(gameCamera != null, "Unable to find GameCamera."))
		{
			assembler.Register<GameCamera>().Allocator(new SingletonAllocator<GameCamera>(gameCamera)).Binding(Binding.Scope);
		}
		AchievementDatabase instance = AssetBundleUtility.LoadAsset<AchievementDatabase>("core", "AchievementDatabase");
		assembler.Register<AchievementDatabase, AchievementDatabase>().Allocator(new SingletonAllocator<AchievementDatabase>(instance)).Binding(Binding.Scope);
		assembler.Register<MenuNavigationAction>().Allocator(new ObjectPool<MenuNavigationAction>
		{
			InitialSize = 5
		});
		assembler.Register<PermanenceZoneTextureLibrary>().Allocator(new GameObjectAllocator<PermanenceZoneTextureLibrary>("core", "PermanenceZoneTextureLibrary")).Binding(Binding.Scope);
		assembler.Register<NotificationScheduler>().Allocator(new HeapAllocator<NotificationScheduler>()).Binding(Binding.Scope);
		if (FeatureToggle.IsFeatureEnabled(Feature.DebugLanguageSwitchButtons) || FeatureToggle.IsFeatureEnabled(Feature.OptionsDebugMenu))
		{
			assembler.Register<DebugLanguageSwitchButtons>().Allocator(new GameObjectPool<DebugLanguageSwitchButtons>("core", "DebugLanguageSwitchButtons")
			{
				InitialSize = 1
			}).Binding(Binding.Scope);
		}
		if (Application.isPlaying && FeatureToggle.IsFeatureEnabled(Feature.OnScreenDebugTools))
		{
			OnScreenDebugToolsActivator onScreenDebugToolsActivator = new GameObject("DebugToolActivator").AddComponent<OnScreenDebugToolsActivator>();
			assembler.Register<OnScreenDebugToolsActivator>().Allocator(new SingletonAllocator<OnScreenDebugToolsActivator>(onScreenDebugToolsActivator)).Binding(Binding.Scope);
			assembler.Register<IDebugRenderSetManager, DebugRenderSetManager>().Allocator(new SingletonAllocator<DebugRenderSetManager>(new DebugRenderSetManager())).Binding(Binding.Scope);
			OnScreenToolManager onScreenToolManager = new GameObject("OnScreenToolManager").AddComponent<OnScreenToolManager>();
			onScreenToolManager.Initialize(onScreenDebugToolsActivator);
			assembler.Register<IOnScreenToolManager, OnScreenToolManager>().Allocator(new SingletonAllocator<OnScreenToolManager>(onScreenToolManager)).Binding(Binding.Scope);
			assembler.Register<OnScreenDebugStorage>().Allocator(new SingletonAllocator<OnScreenDebugStorage>(new OnScreenDebugStorage())).Binding(Binding.Scope);
		}
		else
		{
			assembler.Register<IDebugRenderSetManager, NullDebugRenderSetManager>().Allocator(new SingletonAllocator<NullDebugRenderSetManager>(new NullDebugRenderSetManager())).Binding(Binding.Scope);
			assembler.Register<IOnScreenToolManager, NullOnScreenToolManager>().Allocator(new SingletonAllocator<NullOnScreenToolManager>(new NullOnScreenToolManager())).Binding(Binding.Scope);
		}
		Environment.PopulateAppAssembler(assembler);
		return assembler;
	}

	protected abstract Assembler CreateGameAssembler(Assembler appAssembler);

	private IEnvironment CreateDefaultEnvironment()
	{
		IEnvironment environment = null;
		environment = new WindowsSteamEnvironment();
		if (FeatureToggle.IsFeatureEnabled(Feature.MockPhone))
		{
			environment = new MockEnvironment(environment);
		}
		if (Diagnostics.Verify(environment != null, "We didn't get a default environment for this given platform and variant combination. Is this a new platform or variant?"))
		{
			Log.Info("Using {0} for the current platform and variant.", environment.GetType().ToString());
		}
		return environment;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
