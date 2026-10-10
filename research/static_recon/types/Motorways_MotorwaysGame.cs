using System;
using System.Collections.Generic;
using System.IO;
using Client;
using Factory;
using FixMath;
using GameEvent;
using Motorways.Audio;
using Motorways.Commands;
using Motorways.Leaderboards;
using Motorways.Models;
using Motorways.UI;
using Motorways.Views;
using Plugins.Analytics;
using Server;
using UnityEngine;

namespace Motorways;

public class MotorwaysGame : Game, IGameEventContextProvider
{
	public class MotorwaysGameEvent : global::GameEvent.GameEvent
	{
		public MotorwaysGameEvent(MotorwaysGame game)
		{
			AddContext(game);
		}
	}

	public class MotorwaysGameEndEvent : MotorwaysGameEvent
	{
		public readonly GameEndReason gameEndReason;

		public readonly int score;

		public MotorwaysGameEndEvent(MotorwaysGame game, GameEndReason gameEndReason, int score)
			: base(game)
		{
			this.gameEndReason = gameEndReason;
			this.score = score;
		}
	}

	public class MotorwaysGameStartEvent : MotorwaysGameEvent
	{
		public readonly GameStartReason gameStartReason;

		public MotorwaysGameStartEvent(MotorwaysGame game, GameStartReason gameStartReason)
			: base(game)
		{
			this.gameStartReason = gameStartReason;
		}
	}

	public class MotorwaysGameTimeScaleEvent : MotorwaysGameEvent
	{
		public readonly TimeScale timeScale;

		public readonly TimeScale prevTimeScale;

		public MotorwaysGameTimeScaleEvent(MotorwaysGame game, TimeScale timeScale, TimeScale prevTimeScale)
			: base(game)
		{
			this.timeScale = timeScale;
			this.prevTimeScale = prevTimeScale;
		}
	}

	public class MotorwaysGamePauseEvent : MotorwaysGameTimeScaleEvent
	{
		public readonly bool isPaused;

		public readonly string pauseReason;

		public MotorwaysGamePauseEvent(MotorwaysGame game, bool isPaused, TimeScale timeScale, string pauseReason = "")
			: base(game, timeScale, timeScale)
		{
			this.isPaused = isPaused;
			this.pauseReason = pauseReason;
		}
	}

	public struct MotorwaysGameEventContext : IGameEventContext
	{
		public readonly string cityInstanceId;

		public readonly Fix64 simDays;

		public readonly GameMode gameMode;

		public readonly string cityName;

		public MotorwaysGameEventContext(string cityInstanceId, Fix64 simDays, GameMode gameMode, string cityName)
		{
			this.cityInstanceId = cityInstanceId;
			this.simDays = simDays;
			this.gameMode = gameMode;
			this.cityName = cityName;
		}
	}

	[Dependency]
	private GameEventBroadcaster _eventBroadcaster;

	[Dependency]
	private IAudioSystem _audioSystem;

	[Dependency]
	private City _city;

	[Dependency]
	private MotorwaysThemeDatabase _theme;

	[Dependency]
	private NetworkConnectivityUpdater _connectivityUpdater;

	[Dependency]
	private BuildingPlacer _placer;

	[Dependency]
	private GameCamera _camera;

	[Dependency]
	private Server.Clock _simulationClock;

	[Dependency]
	private LeaderboardService _leaderboardService;

	[Dependency]
	private ActivePlayer _player;

	[Dependency]
	private ChallengeSystem _challengeSystem;

	[Dependency]
	private Pathfinder _pathfinder;

	[Dependency]
	private IAchievementHandler _achievementHandler;

	private readonly IdleVehicleChecker _idleVehicleChecker = new IdleVehicleChecker();

	[Dependency]
	private IDebugRenderSetManager _debugRenderSetManager;

	private GameplayEventHandler _gameplayEventHandler;

	private MotorwaysGameStatistics _lastRecordedStatistics;

	private AchievementStatistics _achievementStatisticsAtStartOfGame;

	private double _nextPulseTime = -1.0;

	private AudioSync _audioSync = new AudioSync();

	private AudioEnvironment _audioEnvironment;

	private int _lastSnapshotFrame;

	private const int SnapshotPeriod = 10;

	private bool _survivedFrame;

	private bool _loggedExceptionReportId;

	private Diagnostics.Report _exceptionReport;

	private bool _hasSubmittedExceptionReport;

	private float _debugTimescale = 1f;

	private int _lastLoggedPlaybackFrame;

	private int _playbackDuration;

	private bool _playingBackSimJournal;

	private float _lastSaveRealTimeSeconds;

	private Fix64 _lastSaveGameTimeHours;

	private const float SaveIntervalRealTimeSeconds = 300f;

	private readonly Fix64 _saveIntervalGameTimeHours = new Fix64(24);

	public IdleVehicleChecker IdleVehicleChecker => _idleVehicleChecker;

	public MapDefinition MapDefinition { get; protected set; }

	public MotorwaysThemeDatabase Theme => _theme;

	public bool HasGameEnded { get; private set; }

	public CityDefinition StartedWithCityDefinition { get; protected set; }

	public GameMode StartedWithGameMode { get; private set; }

	public bool PlayingBackSimJournal => _playingBackSimJournal;

	public float DebugTimescale
	{
		get
		{
			return _debugTimescale;
		}
		set
		{
			_debugTimescale = Mathf.Max(0f, value);
		}
	}

	public void Start(CityDefinition cityDefinition, GameMode mode, MapChallenge mapChallenge, bool replaceExistingRules = false)
	{
		_survivedFrame = true;
		_lastSaveRealTimeSeconds = Time.time;
		_lastSaveGameTimeHours = _simulationClock.Time;
		GameStartReason gameStartReason = GameStartReason.New;
		CityModel model = _simulation.GetModel<CityModel>();
		if (model != null)
		{
			gameStartReason = GameStartReason.Resumed;
		}
		_debugRenderSetManager.Register(cityDefinition);
		Start(gameStartReason);
		HasGameEnded = false;
		StartedWithCityDefinition = cityDefinition;
		StartedWithGameMode = mode;
		if (model != null)
		{
			StartedWithGameMode = model.InitialMode;
		}
		GameRules gameRules;
		if (model != null)
		{
			if (model.Rules == null)
			{
				FixDeserializedSimulation(cityDefinition);
			}
			if (replaceExistingRules)
			{
				gameRules = CreateRulesForMode(base.Scope, mode);
				model.SetGameMode(mode, gameRules);
				_city.Initialize(_city.Definition, gameRules);
			}
			else
			{
				gameRules = model.Rules;
			}
			_placer.SetTileData(_city.Definition.TileWeightData);
		}
		else
		{
			gameRules = CreateRulesForMode(base.Scope, mode);
			if (!TryLoadSimulationJournal(gameRules))
			{
				string cityName = ((MapDefinition != null) ? MapDefinition.cityName : "unknown");
				InitCityCommand command;
				if (mapChallenge == null)
				{
					command = InitCityCommand.CreateNormalCity(base.Scope, cityName, cityDefinition, mode, gameRules, Random.NextSimulationSeed());
				}
				else
				{
					if (mapChallenge.type == MapChallenge.ChallengeType.City)
					{
						mapChallenge.seed = Random.NextSimulationSeed();
					}
					command = InitCityCommand.CreateChallengeCity(base.Scope, cityName, cityDefinition, mode, gameRules, mapChallenge);
				}
				_simulation.ScheduleCommand(command);
			}
		}
		StartAudio();
		_audioSync.StartClock();
		_audioSystem.SignalPulse += OnAudioPulse;
		_simulation.Subscribe(_view);
		_connectivityUpdater.Start();
		MotorwaysClient motorwaysClient = _view as MotorwaysClient;
		GameUIScreen gameUIScreen = base.Scope.Get<GameUIScreen>();
		if (gameRules.ShowsUI())
		{
			gameUIScreen.InitScreen(base.Scope, blocksGameInput: false);
			motorwaysClient.AddView(gameUIScreen);
			motorwaysClient.AddView(base.Scope.Get<HotkeyDebugView>());
			motorwaysClient.AddView(base.Scope.Get<CityScheduleView>());
			motorwaysClient.AddView(base.Scope.Get<TutorialDebugView>());
			motorwaysClient.AddView(base.Scope.Get<IdleVehicleCheckerDebugView>());
			motorwaysClient.AddView(base.Scope.Get<SimulationToggleDebugView>());
		}
		else
		{
			gameUIScreen.gameObject.SetActive(value: false);
		}
		if (gameRules.UseCamera())
		{
			base.Scope.Get<CameraView>().Initialize(gameRules);
		}
		if (gameRules.CanInteract())
		{
			base.Scope.Get<PlayerActionController>().SetGameScope(base.Scope);
		}
		if (gameRules.CanInteract())
		{
			MotorwaysInGameStateToggleController.SwitchToStateIfNeeded(MotorwaysInGameStateToggleController.InGameControllerState.EditingTiles, base.Scope);
		}
		if (gameRules.RecordsGameStatistics())
		{
			_lastRecordedStatistics = base.Scope.Get<MotorwaysGameStatistics>();
			_lastRecordedStatistics.InitFromGame(this);
			_achievementStatisticsAtStartOfGame = new AchievementStatistics();
			if (gameStartReason != GameStartReason.New)
			{
				_achievementStatisticsAtStartOfGame.LogUpgradeStatistics(this, NullAchievementHandler.Instance);
			}
		}
		if (gameRules.CanSave() && gameStartReason == GameStartReason.New && _challengeSystem.GetActiveDailyChallengeSaves(_player).Count > 0)
		{
			MotorwaysTimedChallengeScore challengeScore = _player.GetChallengeScore(MapChallenge.ChallengeType.Daily, _challengeSystem.DailyChallenge.TimeEnd);
			if (challengeScore.ScoreState == LeaderboardScoreState.Editable)
			{
				challengeScore.LockScore();
			}
			LeaderboardId leaderboardId = new DailyLeaderboardId(_challengeSystem.DailyChallenge.TimeStart);
			_leaderboardService.RequestLocalEntry(leaderboardId, delegate(LeaderboardEntry localEntry, long totalLeaderboardEntryCount, LeaderboardError error)
			{
				if (error == null && localEntry != null && localEntry.ScoreState == LeaderboardScoreState.Editable)
				{
					_leaderboardService.SubmitScore(leaderboardId, localEntry.Score, LeaderboardScoreState.Locked);
				}
			});
		}
		if (!(gameRules is BackgroundGameRules))
		{
			_eventBroadcaster.Broadcast(new MotorwaysGameStartEvent(this, gameStartReason));
		}
		if (mode != GameMode.Background)
		{
			_player.Touch();
		}
		_idleVehicleChecker.Initialize(this);
		_gameplayEventHandler = base.Scope.Get<GameplayEventHandler>();
	}

	public bool FixDeserializedSimulation(CityDefinition cityDefinition)
	{
		CityModel model = _simulation.GetModel<CityModel>();
		if (model == null)
		{
			return false;
		}
		GameRules rules = CreateRulesForMode(base.Scope, model.Mode);
		model.SetGameMode(model.Mode, rules);
		if (FeatureToggle.IsFeatureEnabled(Feature.RecordIntersectionDecisions) && _simulation.GetModel<IntersectionDecisionDatabaseModel>() == null)
		{
			IntersectionDecisionDatabaseModel model2 = _simulation.Scope.Get<IntersectionDecisionDatabaseModel>();
			_simulation.AddModel(model2);
		}
		_city.Initialize(cityDefinition, rules);
		return true;
	}

	public void ContinueInMode(GameMode mode)
	{
		GameRules gameRules = CreateRulesForMode(base.Scope, mode);
		HasGameEnded = false;
		_city.SetGameRules(gameRules);
		_simulation.GetModel<CityModel>().SetGameMode(mode, gameRules);
	}

	public override bool TrySave(GameJournalMotive motive)
	{
		if (!_city.Rules.CanSave())
		{
			return false;
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.AppleStoreDemo))
		{
			return false;
		}
		if (HasGameEnded)
		{
			return false;
		}
		MotorwaysGameJournalSave motorwaysGameJournalSave = base.Scope.Get<MotorwaysGameJournalSave>();
		if (!motorwaysGameJournalSave.InitializeFromSimulation(_simulation, motive))
		{
			base.Scope.Release(motorwaysGameJournalSave);
			return false;
		}
		if (motive == GameJournalMotive.AppDeactivated && _survivedFrame)
		{
			StartupScreen.CanAutoResumeDeactivatedGame = true;
		}
		_player.LocalSavedGame = motorwaysGameJournalSave;
		return true;
	}

	public Diagnostics.Report GenerateDiagnosticReport(string motive, DiagnosticReportAttachments attachments)
	{
		Diagnostics.Report report = new Diagnostics.Report();
		report.Motive = motive;
		report.SetMetadata("buildName", Version.Name, index: true);
		report.SetMetadata("buildTimestamp", Version.Timestamp.ToString(), index: true);
		if (!string.IsNullOrEmpty(Version.CommitHash))
		{
			report.SetMetadata("commitHash", Version.CommitHash, index: true);
		}
		try
		{
			List<string> list = new List<string>();
			foreach (Feature value3 in Enum.GetValues(typeof(Feature)))
			{
				if (FeatureToggle.IsFeatureEnabled(value3))
				{
					list.Add(value3.ToString());
				}
			}
			if (list.Count > 0)
			{
				report.SetMetadata("buildFeatures", string.Join(";", list));
			}
		}
		catch (Exception ex)
		{
			Game.Log.Error("Caught exception while trying to attach the list of toggled features to a diagnostic report.\n{0}", ex.ToString());
		}
		try
		{
			ScoreModel model = _simulation.GetModel<ScoreModel>();
			if (model != null)
			{
				if (_city.Rules.ScoringMode == ScoringMode.Trips)
				{
					report.SetMetadata("score", model.Score.ToString());
				}
				else if (_city.Rules.ScoringMode == ScoringMode.EfficiencyMilestones)
				{
					report.SetMetadata("score", model.CurrentEfficiencyMilestone.ToString());
				}
			}
		}
		catch (Exception ex2)
		{
			Game.Log.Error("Caught exception while trying to attach score to a diagnostic report.\n{0}", ex2.ToString());
		}
		try
		{
			ActiveChallengesModel model2 = _simulation.GetModel<ActiveChallengesModel>();
			if (model2.challenges.Count != 0)
			{
				string text = string.Empty;
				foreach (ChallengeData challenge in model2.challenges)
				{
					text = text + challenge.name + ";";
				}
				report.SetMetadata("challenges", text);
				report.SetMetadata("challengesType", model2.challengeType.ToString(), index: true);
			}
		}
		catch (Exception ex3)
		{
			Game.Log.Error("Caught exception while trying to attach challenge info to a diagnostic report.\n{0}", ex3.ToString());
		}
		try
		{
			ClockModel model3 = _simulation.GetModel<ClockModel>();
			if (model3 != null)
			{
				report.SetMetadata("simulationTime", ((float)model3.NextFrame.time).ToString());
			}
		}
		catch (Exception ex4)
		{
			Game.Log.Error("Caught exception while trying to attach time to a diagnostic report.\n{0}", ex4.ToString());
		}
		if (MapDefinition != null)
		{
			report.SetMetadata("city", MapDefinition.mapName, index: true);
		}
		CityModel model4 = _simulation.GetModel<CityModel>();
		if (model4 != null)
		{
			report.SetMetadata("gameMode", model4.Mode.ToString(), index: true);
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.TrackAnalyticsInDiagnosticReports))
		{
			try
			{
				int num = 0;
				int num2 = 0;
				Dictionary<int, int> dictionary = new Dictionary<int, int>();
				ModelListEnumerator<DestinationModel> enumerator3 = _simulation.GetModels<DestinationModel>().GetEnumerator();
				while (enumerator3.MoveNext())
				{
					DestinationModel current2 = enumerator3.Current;
					num2++;
					if (current2.IsOvercrowding)
					{
						num++;
					}
					if (dictionary.ContainsKey(current2.GroupIndex))
					{
						dictionary[current2.GroupIndex]++;
					}
					else
					{
						dictionary.Add(current2.GroupIndex, 1);
					}
				}
				report.SetMetadata("bigPins", num.ToString());
				report.SetMetadata("destinations", num2.ToString());
				DemandModel demandModel = base.Scope.Get<DemandModel>();
				foreach (int key in dictionary.Keys)
				{
					report.SetMetadata($"group{key}Destinations", dictionary[key].ToString());
					int num3 = 0;
					if (demandModel.allocatedPinsInLastWeek.TryGetValue(key, out var value))
					{
						num3 += value.Count;
					}
					report.SetMetadata($"group{key}NewPins", num3.ToString());
				}
				CityPlanModel cityPlanModel = base.Scope.Get<CityPlanModel>();
				for (int i = 0; i < cityPlanModel.groupHouseCounts.Length; i++)
				{
					if (cityPlanModel.groupHouseCounts[i] > 0)
					{
						report.SetMetadata($"group{i}Houses", cityPlanModel.groupHouseCounts[i].ToString());
					}
				}
			}
			catch (Exception ex5)
			{
				Game.Log.Error("Caught exception while trying to attach game statistics to a diagnostic report.\n{0}", ex5.ToString());
			}
			UpgradeDatabaseModel model5 = _simulation.GetModel<UpgradeDatabaseModel>();
			for (int j = 0; j < 9; j++)
			{
				UpgradeType upgradeType = (UpgradeType)j;
				if (model5.numberOfTimesAnUpgradeIsPlaced.TryGetValue(upgradeType, out var value2) || model5.timesUpgradePresented[j] > 0)
				{
					report.SetMetadata($"upgrade{upgradeType}_currentlyUsed", (model5.GetTotalUpgradeCount(upgradeType) - model5.GetAvailableUpgradeCount(upgradeType)).ToString());
					report.SetMetadata($"upgrade{upgradeType}_totalAwarded", model5.GetTotalUpgradeCount(upgradeType).ToString());
					report.SetMetadata($"upgrade{upgradeType}_timesPlaced", value2.ToString());
					report.SetMetadata($"upgrade{upgradeType}_presented", model5.timesUpgradePresented[(int)upgradeType].ToString());
					report.SetMetadata($"upgrade{upgradeType}_packagesTaken", model5.NumberOfPackagesTakenOf(upgradeType).ToString());
				}
			}
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.RecordAppJournal) && attachments.HasFlag(DiagnosticReportAttachments.AppCommandJournal))
		{
			IScope scope = base.Scope.Get<App>().Scope;
			AppCommandJournal obj = base.Scope.Get<AppCommandJournal>();
			MemoryStream memoryStream = new MemoryStream();
			using (BinaryWriter writer = new BinaryWriter(memoryStream))
			{
				scope.Export(obj, writer);
			}
			report.AttachFile("commands.appjournal", memoryStream.ToArray());
		}
		if (attachments.HasFlag(DiagnosticReportAttachments.SimCommandJournal))
		{
			CommandJournal commandJournal = base.Scope.Get<CommandJournal>();
			if (commandJournal.EntryCount > 0)
			{
				MemoryStream memoryStream2 = new MemoryStream();
				using (BinaryWriter writer2 = new BinaryWriter(memoryStream2))
				{
					base.Scope.Export(commandJournal, writer2);
				}
				report.AttachFile("commands.simjournal", memoryStream2.ToArray());
			}
		}
		if (attachments.HasFlag(DiagnosticReportAttachments.SimArchive) && _city.Rules.CanSave())
		{
			MotorwaysGameJournalSave motorwaysGameJournalSave = base.Scope.Get<MotorwaysGameJournalSave>();
			if (motorwaysGameJournalSave.InitializeFromSimulation(_simulation, GameJournalMotive.DiagnosticsReport))
			{
				MemoryStream memoryStream3 = new MemoryStream();
				using (BinaryWriter binaryWriter = new BinaryWriter(memoryStream3))
				{
					motorwaysGameJournalSave.OnSerializeBeforeData(binaryWriter);
					binaryWriter.Write(motorwaysGameJournalSave.GetBytesForSerializing());
				}
				report.AttachFile("simulation.gamejournal", memoryStream3.ToArray());
			}
			base.Scope.Release(motorwaysGameJournalSave);
		}
		if (attachments.HasFlag(DiagnosticReportAttachments.Log) && FeatureToggle.IsFeatureEnabled(Feature.RecordLogs))
		{
			byte[] recordedLog = Diagnostics.Log.RecordedLog;
			if (recordedLog != null)
			{
				report.AttachFile("log.txt", recordedLog);
			}
		}
		if (attachments.HasFlag(DiagnosticReportAttachments.Screenshot))
		{
			_camera.DisableAllBlur();
			GameObject gameObject = new GameObject();
			Camera camera = gameObject.AddComponent<Camera>();
			camera.CopyFrom(_camera.DefaultCamera);
			Fix64 cameraSizeAtTime = _city.GetCameraSizeAtTime(base.Scope.Get<ClockModel>().NextFrame.time);
			RectFixed clientPlayableAreaAtZoom = _city.GetClientPlayableAreaAtZoom(cameraSizeAtTime);
			Vector3 position = new Vector3((float)clientPlayableAreaAtZoom.Center.x, (float)clientPlayableAreaAtZoom.Center.y, camera.transform.position.z);
			camera.transform.position = position;
			camera.orthographicSize = base.Scope.Get<CameraView>().MaxZoom;
			DelegateCanvasGroup component = base.Scope.Get<GameUIScreen>().GetComponent<DelegateCanvasGroup>();
			float alpha = component.Alpha;
			component.Alpha = 0f;
			((MotorwaysThemeDatabase)_themeDatabase).materialCollection.SetWorldGridThickness(0f);
			float num4 = Mathf.Min(1f, 1024f / (float)Mathf.Max(Screen.width, Screen.height));
			RenderTexture temporary = RenderTexture.GetTemporary(Mathf.RoundToInt((float)Screen.width * num4), Mathf.RoundToInt((float)Screen.height * num4), 24, RenderTextureFormat.ARGB32);
			RenderTexture active = RenderTexture.active;
			RenderTexture.active = temporary;
			camera.targetTexture = temporary;
			camera.Render();
			Texture2D texture2D = new Texture2D(camera.targetTexture.width, camera.targetTexture.height, TextureFormat.RGB24, mipChain: false);
			texture2D.ReadPixels(new Rect(0f, 0f, camera.targetTexture.width, camera.targetTexture.height), 0, 0);
			texture2D.Apply();
			byte[] data = texture2D.EncodeToJPG();
			report.AttachFile("screenshot.jpg", data);
			RenderTexture.active = active;
			UnityEngine.Object.Destroy(texture2D);
			UnityEngine.Object.Destroy(gameObject);
			RenderTexture.ReleaseTemporary(temporary);
			component.Alpha = alpha;
			_camera.ReEnableAllBlur();
		}
		return report;
	}

	public override void StopAudio()
	{
		string text = ((StartedWithCityDefinition != null) ? StartedWithCityDefinition.name : "unknown");
		string message = text + ".StopAudio() : AudioEnvironment has already been nuked. Skipping ...";
		if (_audioEnvironment != null)
		{
			_audioEnvironment.Kill();
			_audioEnvironment = null;
			message = text + ".StopAudio() : Success. Killing AudioEnvironment.";
		}
		Dbug.Log.Info(message);
	}

	public void ClearPathfinder()
	{
		_pathfinder.Clear();
	}

	public void PausePathfinder()
	{
		_pathfinder.PauseUpdate();
	}

	public void ResumePathfinder()
	{
		_pathfinder.ResumeUpdate();
	}

	public void StartAudio()
	{
		if (_audioEnvironment != null && _audioEnvironment.Active)
		{
			Dbug.Log.Info(StartedWithCityDefinition.name + ".StartAudio(): AudioEnvironment is already active. Skipping ...");
			return;
		}
		AudioLoadout audioLoadout = null;
		if (StartedWithCityDefinition.audioLoadout != null)
		{
			Dbug.Log.Info(StartedWithCityDefinition.name + ".StartAudio() : Refreshing City Loadout.");
			audioLoadout = _audioSystem.GetLoadout(StartedWithCityDefinition.audioLoadout.name);
		}
		if (audioLoadout != null)
		{
			Dbug.Log.Info(StartedWithCityDefinition.name + ".StartAudio() : Activate Audio Environment With a New Loadout + City.");
			_audioEnvironment = new AudioEnvironment(audioLoadout, _city, this);
		}
	}

	public void SetMapDefinition(MapDefinition newMapDefinition)
	{
		MapDefinition = newMapDefinition;
	}

	public override void OnGameEnd(GameEndReason gameEndReason)
	{
		base.OnGameEnd(gameEndReason);
		SetPaused(isPaused: true, AnalyticsTypes.PauseReason.GameEnd);
		_themeDatabase.DisableDeleteModeOverrides();
		if (HasGameEnded)
		{
			Diagnostics.FailAssert("A game can only be ended once.");
			return;
		}
		HasGameEnded = true;
		if (gameEndReason == GameEndReason.GameOver)
		{
			DeleteLocalSave();
			base.Scope.Get<CameraView>().ResetPlayerViewport();
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.SubmitDiagnosticReportOnGameOver))
		{
			UploadDiagnosticsReport(gameEndReason);
		}
		GetSimDays();
		if (StartedWithGameMode == GameMode.Tutorial && gameEndReason == GameEndReason.Exit)
		{
			gameEndReason = GameEndReason.TutorialSkipped;
		}
		_eventBroadcaster.Broadcast(new MotorwaysGameEndEvent(this, gameEndReason, base.Scope.Get<ScoreModel>().Score));
		if (_simulation.Scope.Get<ScoreModel>().Score != 0)
		{
			GameRules rules = _city.Rules;
			if (rules.RecordsGameStatistics())
			{
				RecordGameStatistics(gameEndReason);
			}
			if (rules.SupportsLeaderboards())
			{
				UpdateLeaderboardIfRequired(gameEndReason);
			}
			if (gameEndReason == GameEndReason.GameOver)
			{
				_player.MotorwaysExtendedUserProfile.GameOverCount++;
			}
		}
		else
		{
			DeleteLocalSave();
		}
		_player.Touch();
		_debugRenderSetManager.Unregister(StartedWithCityDefinition);
	}

	private void UpdateLeaderboardIfRequired(GameEndReason gameEndReason)
	{
		ActiveChallengesModel activeChallengesModel = _simulation.Scope.Get<ActiveChallengesModel>();
		if (activeChallengesModel.challengeType != MapChallenge.ChallengeType.Mystery)
		{
			bool flag = activeChallengesModel.HasChallenges && (activeChallengesModel.challengeType == MapChallenge.ChallengeType.Daily || activeChallengesModel.challengeType == MapChallenge.ChallengeType.Weekly);
			if ((!flag || activeChallengesModel.IsActiveWithGracePeriod) && (flag || gameEndReason == GameEndReason.GameOver))
			{
				int score = base.Scope.Get<ScoreModel>().Score;
				LeaderboardScoreState scoreState = ((!MotorwaysScoreValidation.ShouldLockScoreWhenGameEnds(activeChallengesModel.challengeType, gameEndReason)) ? LeaderboardScoreState.Editable : LeaderboardScoreState.Locked);
				_leaderboardService.SubmitScore(GetLeaderboardIdForGame(), score, scoreState);
			}
		}
	}

	public override void OnReleasedFromScope(IScope scope)
	{
		base.OnReleasedFromScope(scope);
		if (_lastRecordedStatistics != null)
		{
			base.Scope.Release(_lastRecordedStatistics);
			_lastRecordedStatistics = null;
		}
	}

	private void DeleteLocalSave()
	{
		if (_player.HasLocalSavedGame)
		{
			Game.Log.Info("Deleting local save after ending the game.");
			_player.LocalSavedGame = null;
		}
	}

	private void UploadDiagnosticsReport(GameEndReason gameEndReason)
	{
		string text = gameEndReason.ToString();
		if (text.Length >= 2)
		{
			text = char.ToLower(text[0]) + text.Substring(1);
		}
		GenerateDiagnosticReport(text, DiagnosticReportAttachments.SimCommandJournal | DiagnosticReportAttachments.Screenshot).Upload();
	}

	public void RecordGameStatistics(GameEndReason? gameEndReason = null)
	{
		if (!_city.Rules.RecordsGameStatistics())
		{
			return;
		}
		MotorwaysGameStatistics motorwaysGameStatistics = base.Scope.Get<MotorwaysGameStatistics>();
		motorwaysGameStatistics.InitFromGameIncrementally(this, _lastRecordedStatistics, gameEndReason);
		_player.RecordGameStatistics(motorwaysGameStatistics);
		if (_city.Rules.ScoringMode == ScoringMode.Trips)
		{
			_player.AchievementStatistics.LogScoreStatistics(motorwaysGameStatistics, _achievementHandler);
		}
		_player.AchievementStatistics.LogUpgradeStatistics(this, _achievementHandler, _achievementStatisticsAtStartOfGame);
		if (gameEndReason.HasValue && gameEndReason == GameEndReason.GameOver)
		{
			_player.AchievementStatistics.LogGameOverStatistics(this, _achievementHandler);
		}
		_player.CheckLifetimeAchievements();
		if (_lastRecordedStatistics != null)
		{
			base.Scope.Release(_lastRecordedStatistics);
		}
		_lastRecordedStatistics = motorwaysGameStatistics;
		ActiveChallengesModel activeChallengesModel = base.Scope.Get<ActiveChallengesModel>();
		if (!activeChallengesModel.HasChallenges || activeChallengesModel.challengeType != MapChallenge.ChallengeType.City)
		{
			return;
		}
		int challengeIndex = -1;
		for (int i = 0; i < MapDefinition.cityChallenges.Length; i++)
		{
			bool flag = true;
			if (MapDefinition.cityChallenges[i].challenges.Length != activeChallengesModel.challenges.Count)
			{
				continue;
			}
			ChallengeData[] challenges = MapDefinition.cityChallenges[i].challenges;
			foreach (ChallengeData item in challenges)
			{
				if (!activeChallengesModel.challenges.Contains(item))
				{
					flag = false;
					break;
				}
			}
			if (flag)
			{
				challengeIndex = i;
			}
		}
		CityChallengeStatistics cityChallengeScore = _player.GetCityChallengeScore(MapDefinition.cityName, GameMode.Normal, challengeIndex);
		int score = base.Scope.Get<ScoreModel>().Score;
		if (cityChallengeScore.BestScore < score)
		{
			cityChallengeScore.BestScore = score;
		}
	}

	public override void Tick(float frameTime)
	{
		if (!_survivedFrame)
		{
			StartupScreen.CanAutoResumeDeactivatedGame = false;
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.SubmitDiagnosticReportOnException))
		{
			bool flag = !_survivedFrame && _exceptionReport == null;
			if (FeatureToggle.IsFeatureEnabled(Feature.SubmitOnlyOneDiagnosticReportOnExceptionPerGame))
			{
				flag &= !_hasSubmittedExceptionReport;
			}
			if (flag)
			{
				_exceptionReport = GenerateDiagnosticReport("exception", DiagnosticReportAttachments.SimCommandJournal | DiagnosticReportAttachments.Log);
				if (!string.IsNullOrEmpty(Diagnostics.Exception.LastException))
				{
					_exceptionReport.SetMetadata("exception", Diagnostics.Exception.LastException);
					Diagnostics.Exception.LastException = null;
				}
				if (!string.IsNullOrEmpty(Diagnostics.Exception.LastExceptionStackTrace))
				{
					_exceptionReport.SetMetadata("stackTrace", Diagnostics.Exception.LastExceptionStackTrace);
					Diagnostics.Exception.LastExceptionStackTrace = null;
				}
				_exceptionReport.Upload();
				_hasSubmittedExceptionReport = true;
			}
			if (_exceptionReport != null && _exceptionReport.Id >= 0 && !_loggedExceptionReportId)
			{
				_loggedExceptionReportId = true;
				Debug.LogFormat("Caught exception during MotorwaysGame.Tick() and submitted report with id {0}.", _exceptionReport.Id);
			}
		}
		_survivedFrame = false;
		if (_playingBackSimJournal)
		{
			IInputState inputState = base.Scope.Get<IInputState>();
			if (_simulation.HasAnyScheduledCommands)
			{
				inputState.BlockActions = true;
				int frameCount = base.Scope.Get<Server.Clock>().FrameCount;
				if (frameCount - _lastLoggedPlaybackFrame >= 25)
				{
					_lastLoggedPlaybackFrame = frameCount;
					Game.Log.Info("Journal playback up to simulation frame {0} / {1}.", frameCount, _playbackDuration);
				}
			}
			else
			{
				inputState.BlockActions = false;
				_playingBackSimJournal = false;
				Game.Log.Info("Completed journal playback, switching to standard execution.");
			}
		}
		else if (StartedWithGameMode != GameMode.Background && FeatureToggle.IsFeatureEnabled(Feature.ValidateSimulationDeterminism) && !_simulation.IsPaused && _simulationClock.FrameCount - _lastSnapshotFrame > 10)
		{
			_lastSnapshotFrame = _simulationClock.FrameCount;
			_simulation.ScheduleCommand(base.Scope.Get<SnapshotCommand>());
		}
		base.Tick(frameTime);
		_idleVehicleChecker.RunCheck();
		if (_playingBackSimJournal && _simulation.IsPaused && _simulation.HasAnyScheduledCommands)
		{
			int frameIndex = _simulation.NextScheduledCommand.FrameIndex;
			while (frameIndex > _simulationClock.FrameCount + 1)
			{
				base.Tick(frameTime);
			}
		}
		_connectivityUpdater.Tick();
		if (_audioEnvironment != null)
		{
			_audioEnvironment.Update();
		}
		_survivedFrame = true;
		if (_city.Rules.ShouldSavePeriodically)
		{
			Fix64 fix = _simulationClock.Time / (Fix64)(5.0 / 6.0);
			if (Time.time - _lastSaveRealTimeSeconds > 300f || fix - _lastSaveGameTimeHours > _saveIntervalGameTimeHours)
			{
				TrySave(GameJournalMotive.Autosave);
				_lastSaveRealTimeSeconds = Time.time;
				_lastSaveGameTimeHours = fix;
			}
		}
		_gameplayEventHandler.Tick(this);
	}

	public void TickDuringTransition(float frameTime)
	{
		CameraView cameraView = base.Scope.Get<ViewClient>().CameraView;
		if (cameraView != null)
		{
			_timeInterval.UnsyncedDelta = frameTime;
			_timeInterval.Delta = frameTime;
			cameraView.Tick(_timeInterval, 0f);
		}
	}

	public static GameRules CreateRulesForMode(IScope gameScope, GameMode mode)
	{
		return mode switch
		{
			GameMode.Tutorial => gameScope.Get<TutorialGameRules>(), 
			GameMode.Background => gameScope.Get<BackgroundGameRules>(), 
			GameMode.Endless => gameScope.Get<EndlessGameRules>(), 
			GameMode.Expert => gameScope.Get<ExpertGameRules>(), 
			GameMode.Creative => gameScope.Get<CreativeGameRules>(), 
			GameMode.Movie => gameScope.Get<MovieGameRules>(), 
			GameMode.Cinematic => gameScope.Get<CinematicGameRules>(), 
			_ => gameScope.Get<GameRules>(), 
		};
	}

	protected override void AdjustTimeInterval(TimeInterval timeInterval)
	{
		timeInterval.UnsyncedDelta *= _debugTimescale;
		timeInterval.Delta *= _debugTimescale;
		_audioSync.SyncTimeInterval(timeInterval, _nextPulseTime, _audioSystem);
	}

	private void OnAudioPulse(double pulseTime, int pulseIndex, int pulseLoopCount)
	{
		_nextPulseTime = pulseTime;
	}

	public override bool CanInteract()
	{
		City city = _simulation.Scope.Get<City>();
		if (city != null && base.CanInteract())
		{
			GameRules rules = city.Rules;
			if (rules != null)
			{
				return rules.CanInteract();
			}
		}
		Diagnostics.FailAssert("We should never get here!");
		return false;
	}

	private bool TryLoadSimulationJournal(GameRules rules)
	{
		_playingBackSimJournal = false;
		_playbackDuration = -1;
		_lastLoggedPlaybackFrame = 0;
		if (StartedWithGameMode == GameMode.Normal)
		{
			string text = null;
			if (FeatureToggle.IsFeatureEnabled(Feature.SoakTest))
			{
				text = $"{Application.streamingAssetsPath}/SoakTestJournals/{MapDefinition.cityName}.bytes";
			}
			else if (Application.isEditor)
			{
				text = UnityEngine.Object.FindObjectOfType<AppRuntime>()?._playbackSimJournalPath;
			}
			if (!string.IsNullOrEmpty(text) && File.Exists(text))
			{
				CommandJournal commandJournal = null;
				using (BinaryReader reader = new BinaryReader(File.Open(text, FileMode.Open, FileAccess.Read)))
				{
					commandJournal = base.Scope.Import<CommandJournal>(reader);
				}
				bool flag = false;
				if (commandJournal != null)
				{
					for (int i = 0; i < commandJournal.EntryCount; i++)
					{
						Command entry = commandJournal.GetEntry(i);
						if (entry is InitCityCommand)
						{
							InitCityCommand initCityCommand = entry as InitCityCommand;
							if (MapDefinition == null || MapDefinition.cityName != initCityCommand.CityName)
							{
								Game.Log.Warn("Not loading simulation command journal; it is for {0}, but this game has loaded {1}.", initCityCommand.CityName, (MapDefinition != null) ? MapDefinition.cityName : "unknown");
							}
							else
							{
								initCityCommand.Rules = rules;
								initCityCommand.CityDefinition = StartedWithCityDefinition;
								flag = true;
							}
							break;
						}
					}
					if (flag)
					{
						for (int j = 0; j < commandJournal.EntryCount; j++)
						{
							Command entry2 = commandJournal.GetEntry(j);
							_simulation.ScheduleCommand(entry2);
							_playbackDuration = Mathf.Max(_playbackDuration, entry2.FrameIndex);
						}
						commandJournal.Clear();
						if (FeatureToggle.IsFeatureEnabled(Feature.SoakTest))
						{
							Command command = SetPausedCommand.Create(base.Scope, pause: false);
							command.FrameIndex = _playbackDuration + 1;
							_simulation.ScheduleCommand(command);
						}
						_playingBackSimJournal = true;
						return true;
					}
					for (int k = 0; k < commandJournal.EntryCount; k++)
					{
						base.Scope.Release(commandJournal.GetEntry(k));
					}
					commandJournal.Clear();
					Game.Log.Warn("Unable to find InitCityCommand in simulation command journal.");
				}
				else
				{
					Game.Log.Warn("Unable to deserialise simulation command journal from file {0}.", text);
				}
			}
		}
		return false;
	}

	public LeaderboardId GetLeaderboardIdForGame()
	{
		ActiveChallengesModel activeChallengesModel = base.Scope.Get<ActiveChallengesModel>();
		if (activeChallengesModel.HasChallenges)
		{
			if (activeChallengesModel.challengeType == MapChallenge.ChallengeType.Daily)
			{
				return new DailyLeaderboardId(activeChallengesModel.timeStart);
			}
			if (activeChallengesModel.challengeType == MapChallenge.ChallengeType.Weekly)
			{
				return new WeeklyLeaderboardId(activeChallengesModel.timeStart);
			}
			if (activeChallengesModel.challengeType == MapChallenge.ChallengeType.City)
			{
				return new CityLeaderboardId(MapDefinition.CityNameEnum, CityGameMode.CityChallenge, activeChallengesModel.cityChallengeIndex);
			}
			Diagnostics.FailAssert("Invalid challenge type for leaderboardId: {0}", activeChallengesModel.challengeType);
			return null;
		}
		CityGameMode mode = ((StartedWithGameMode == GameMode.Expert) ? CityGameMode.Expert : CityGameMode.Normal);
		return new CityLeaderboardId(MapDefinition.CityNameEnum, mode, -1);
	}

	protected Fix64 GetSimDays()
	{
		Fix64 result = Fix64.Zero;
		if (_simulation != null)
		{
			ClockModel model = _simulation.GetModel<ClockModel>();
			if (model != null)
			{
				result = model.FractionalDays;
			}
		}
		return result;
	}

	protected override void OnTimeScaleChanged(TimeScale prevScale, TimeScale scale)
	{
		base.OnTimeScaleChanged(prevScale, scale);
		GetSimDays();
		_eventBroadcaster.Broadcast(new MotorwaysGameTimeScaleEvent(this, scale, prevScale));
	}

	protected override void OnPauseChanged(bool isPaused, TimeScale timeScale, string pauseReason = "")
	{
		base.OnPauseChanged(isPaused, timeScale);
		GetSimDays();
		_eventBroadcaster.Broadcast(new MotorwaysGamePauseEvent(this, isPaused, timeScale, pauseReason));
	}

	public IGameEventContext ProvideGameEventContext()
	{
		Fix64 simDays = GetSimDays();
		return new MotorwaysGameEventContext(_city.InstanceId, simDays, _city.GameMode, MapDefinition?.cityName ?? _city.Definition.name);
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
