using System;
using System.IO;
using System.Threading.Tasks;
using Factory;
using Factory.Allocators;
using Motorways;
using Motorways.Views;
using UnityEngine;

public class AppRuntime : MonoBehaviour
{
	[HideInInspector]
	public string _playbackAppJournalPath;

	[HideInInspector]
	public string _playbackSimJournalPath;

	[HideInInspector]
	public string _recordingAppJournalPath;

	private BaseInputOverride _inputOverride;

	private InputModule _inputModule;

	private AppContainer _container;

	private DeepLinkProcessor _deepLinkProcessor;

	private ScreenStack _screenStack;

	private static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("AppRuntime");

	public IApp App => _container?.App;

	public bool CanExportAppJournal
	{
		get
		{
			if (_container.CommandJournal != null)
			{
				return !string.IsNullOrEmpty(_recordingAppJournalPath);
			}
			return false;
		}
	}

	private void Awake()
	{
		Diagnostics.IsTrackingExceptions = true;
	}

	private void OnApplicationPause(bool pauseStatus)
	{
		if (!pauseStatus)
		{
			App?.GameOpenedNotificationSetup();
		}
	}

	private void OnApplicationFocus(bool hasFocus)
	{
		if (!hasFocus)
		{
			App?.Scope.Get<ScreenStack>().OnApplicationPaused();
		}
	}

	private void SetupInputOverride()
	{
		_inputOverride = AppContainer.Environment.AddInputOverrideToGameObject(base.gameObject);
		_inputModule = base.gameObject.AddComponent<InputModule>();
		_inputModule.inputOverride = _inputOverride;
		_inputModule.horizontalAxis = "";
		_inputModule.verticalAxis = "";
		_inputModule.submitButton = "";
		_inputModule.cancelButton = "";
	}

	private void Start()
	{
		FeatureToggle.RemoveAllSources();
		if (Application.isEditor)
		{
			FeatureToggle.AddSource(new EditorPrefsConfigSettingSource());
		}
		if (_container == null)
		{
			_container = new MotorwaysAppContainer();
		}
		_container.CreateAssemblers();
		if (FeatureToggle.IsFeatureEnabled(Feature.OptionsDebugMenu))
		{
			FeatureToggle.AddSource(new OptionsMenuSettingSource());
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.RecordLogs))
		{
			Diagnostics.Log.IsRecordingLog = true;
		}
		SetupInputOverride();
		bool flag = !string.IsNullOrEmpty(_playbackAppJournalPath);
		if (flag)
		{
			_container.AppAssembler.Register<IAppCommandSource, JournalAppCommandSource>().Allocator(new HeapAllocator<JournalAppCommandSource>()).Binding(Binding.Scope);
		}
		_container.CreateScope();
		if (flag)
		{
			using BinaryReader reader = new BinaryReader(File.Open(_playbackAppJournalPath, FileMode.Open));
			_container.AppScope.Import(reader);
		}
		else if (Diagnostics.File.CanWrite)
		{
			DateTime now = DateTime.Now;
			_recordingAppJournalPath = Diagnostics.File.GetFullPath($"{now.Year:D4}{now.Month:D2}{now.Day:D2}{now.Hour:D2}{now.Minute:D2}.appjournal");
		}
		IApp app = _container.CreateApp();
		_inputOverride.InputState = app.InputState;
		bool recordJournal = false;
		if (FeatureToggle.IsFeatureEnabled(Feature.RecordAppJournal))
		{
			recordJournal = !flag;
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.ElevateErrorsForCloudDiagnostics))
		{
			Log.Error("Cloud diagnostics enabled.");
		}
		_container.Start(recordJournal);
		_deepLinkProcessor = App?.Scope.Get<DeepLinkProcessor>();
		_screenStack = App?.Scope.Get<ScreenStack>();
	}

	private void Update()
	{
		_container.Tick();
		CheckForDeepLinkChallenge();
	}

	public void ExportAppJournal()
	{
		using (BinaryWriter writer = new BinaryWriter(File.Open(_recordingAppJournalPath, FileMode.Create)))
		{
			_container.AppScope.Export(_container.CommandJournal, writer);
		}
		Log.Info("Exported journal to {0}.", _recordingAppJournalPath);
	}

	private void CheckForDeepLinkChallenge()
	{
		if (_deepLinkProcessor != null && _screenStack != null && !_screenStack.ExitingToMainMenu && _deepLinkProcessor.hasChallengeToUse && _screenStack.HasVisibleScreens() && _screenStack.IsScreenInStack<MainMenuScreen>())
		{
			HandleDeeplinkRequest(_screenStack, _deepLinkProcessor);
		}
	}

	private async Task HandleDeeplinkRequest(ScreenStack screenStack, DeepLinkProcessor deepLinkProcessor)
	{
		Diagnostics.Log.Info("DeepLinkProcessor", "Handling deeplink");
		if (screenStack.GetTopActiveScreenType() != ScreenStack.MotorwaysScreen.MainMenu)
		{
			await screenStack.ExitToMainMenu();
		}
		screenStack.PushScreen(ScreenStack.MotorwaysScreen.MapSelect, delegate(MapSelectScreen screen)
		{
			screen.PrepareScreen(null, handleDeeplinkChallenge: true, changeBlurWhenTransitioning: true);
		});
		deepLinkProcessor.hasChallengeToUse = false;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
