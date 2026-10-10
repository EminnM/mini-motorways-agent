using System.Collections.Generic;
using Client;
using Easing;
using Factory;
using FixMath;
using Motorways.Models;
using Motorways.Processes;
using NaughtyAttributes;
using NotificationService.Events;
using Plugins.Analytics;
using Screens;
using Server;
using SoftwareCapabilities;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

namespace Motorways.Views;

public class GameContainerScreen : BaseScalingScreen, IGameStartScreen
{
	[Dependency]
	private VisualConstantsData _constants;

	[Dependency]
	protected PlayerActionController _playerActionController;

	[Dependency]
	private ISoftwareCapabilities _softwareCapabilities;

	[Dependency]
	private IAppCommandSource _runtimeAppCommandSource;

	private MotorwaysGame _game;

	private bool _startGameOnTransition;

	private CityDefinition _newCity;

	private MapDefinition _newMapDefinition;

	private MapChallenge _newMapChallenge;

	private GameUIScreen _gameUIScreen;

	private CameraView _cameraView;

	private GameMode _gameMode;

	private bool _playerPausedGame;

	private bool _startPaused;

	private bool _gameSuspended;

	private bool _recentlyExitedCinematicMode;

	[MaxValue(0.9999f)]
	[SerializeField]
	[MinValue(0.001f)]
	private float _percentageOfTransitionInToStartBluringForChallenges = 0.6f;

	private bool _hasSeenChallenges;

	private bool _hasSeenModeInfo;

	private bool _isTransitioningOutToBeReleased;

	private bool _overrideTransitionInAnimation;

	[Dependency]
	private MenuPlacementDefinition _menuPlacementDefinition;

	[Dependency]
	private INotificationEventSystem _notificationEventSystem;

	private static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("GameContainerScreen");

	private static readonly ProfilerMarker Profiler_Tick = new ProfilerMarker(ProfilerCategory.Scripts, "GameContainerScreen.Tick()");

	public string CurrentCityName => _newMapDefinition?.cityName;

	public override void Reset()
	{
		base.Reset();
		_hasSeenChallenges = false;
		_hasSeenModeInfo = false;
		_startGameOnTransition = false;
		_gameMode = GameMode.Normal;
		_playerPausedGame = false;
		_isTransitioningOutToBeReleased = false;
		_overrideTransitionInAnimation = false;
		_startPaused = false;
		_recentlyExitedCinematicMode = false;
		_gameSuspended = false;
	}

	public virtual void PrepareForMap(CityDefinition newCity, MapDefinition newMapDefinition, GameMode gameMode, MapChallenge newMapChallenge = null, bool startPaused = false)
	{
		if (_game != null || _newCity != null)
		{
			Log.Warn("We have not properly cleaned up the last game! It is possible the game container screen has not been disposed quickly enough. Cleaning up the game now.");
			CleanupPreviousGame();
		}
		_newCity = newCity;
		_newMapDefinition = newMapDefinition;
		_gameMode = gameMode;
		_newMapChallenge = newMapChallenge;
		_startGameOnTransition = true;
		_overrideTransitionInAnimation = true;
		_startPaused = startPaused;
		RegisterThemeComponents(_themeDatabase.GetTheme());
		_ = newMapChallenge?.type;
		_ = newMapChallenge?.cityChallengeIndex;
		if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore)
		{
			ParticleSystem[] componentsInChildren = newCity.GetComponentsInChildren<ParticleSystem>();
			for (int i = 0; i < componentsInChildren.Length; i++)
			{
				componentsInChildren[i].gameObject.SetActive(value: false);
			}
		}
	}

	public virtual void PrepareForNewGame(CityDefinition newCity, MapDefinition newMapDefinition, MotorwaysGame game, MapChallenge newMapChallenge = null, bool startPaused = false)
	{
		if (_game != null || _newCity != null)
		{
			Log.Warn("We have not properly cleaned up the last game! It is possible the game container screen has not been disposed quickly enough. Cleaning up the game now.");
			CleanupPreviousGame();
		}
		_game = game;
		_newCity = newCity;
		_newMapDefinition = newMapDefinition;
		_gameMode = (_game.Simulation?.GetModel<CityModel>()?.Mode).GetValueOrDefault();
		_newMapChallenge = newMapChallenge;
		_startGameOnTransition = true;
		_overrideTransitionInAnimation = true;
		_startPaused = startPaused;
		RegisterThemeComponents(_themeDatabase.GetTheme());
		if (_newCity.bonusTreeGrassObjects != null)
		{
			bool usesBonusTrees = _game.Simulation.GetModel<GameBehaviourModel>().UsesBonusTrees;
			GameObject[] bonusTreeGrassObjects = _newCity.bonusTreeGrassObjects;
			for (int i = 0; i < bonusTreeGrassObjects.Length; i++)
			{
				bonusTreeGrassObjects[i].SetActive(usesBonusTrees);
			}
		}
	}

	public override void RegisterThemeComponents(ITheme theme)
	{
		base.RegisterThemeComponents(theme);
		if (!(_newCity != null))
		{
			return;
		}
		List<IThemeComponent> list = new List<IThemeComponent>();
		_newCity.GetComponentsInChildren(includeInactive: true, list);
		if (list != null)
		{
			foreach (IThemeComponent item in list)
			{
				item.InitializeTheme(_themeDatabase);
			}
		}
		if (themeComponents == null)
		{
			themeComponents = list;
		}
		else
		{
			themeComponents.AddRange(list);
		}
	}

	public override void ScaleToCamera()
	{
		ScaleToGameCamera();
	}

	public virtual void PrepareForRestartMap(GameMode gameMode)
	{
		_gameMode = gameMode;
		ReleaseGame();
		_ = _newMapChallenge?.type;
		_ = _newMapChallenge?.cityChallengeIndex;
		_startGameOnTransition = true;
	}

	public virtual void PrepareForContinueInMode(GameMode gameMode)
	{
		_startGameOnTransition = false;
		_gameMode = gameMode;
		ModelListEnumerator<RoundaboutModel> enumerator = _game.Simulation.GetModels<RoundaboutModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			RoundaboutModel current = enumerator.Current;
			if (current.CenterTileModel.Tile.IsRoundaboutPermanent)
			{
				current.RestoreConcreteFromStoredReplacedConnections(RoundaboutModel.ConcreteRestoreType.Release);
			}
			current.ClearReplacedConnections();
		}
		_game.ContinueInMode(gameMode);
		_playerActionController.SetGameScope(_game.Scope);
		_gameUIScreen.ScoreView.SetupView();
		_hasSeenModeInfo = false;
		_game.Simulation.GetModel<ScoreModel>().OnContinuedInMode(gameMode);
		switch (gameMode)
		{
		case GameMode.Endless:
			_game.Scope.Get<ActiveChallengesModel>().RemoveChallengesForEndless();
			_gameUIScreen.UpgradeBar.RefreshAllAvailableUpgradeStacks();
			break;
		case GameMode.Creative:
		{
			UpgradeDatabaseModel upgradeDatabaseModel = _game.Scope.Get<UpgradeDatabaseModel>();
			upgradeDatabaseModel.AwardStartingPackages();
			_game.Scope.Get<ClientUpgradeDatabase>().OnUpgradesChanged(upgradeDatabaseModel);
			_game.Scope.Get<CityPlanModel>().ClearScheduledBuildings(_game.Scope);
			_game.Scope.Get<ClockModel>().SetExpansionTimeToDay(_game.Scope.Get<City>().Definition.cameraZoom.durationInDays);
			ModelListEnumerator<DestinationModel> enumerator2 = _game.Simulation.GetModels<DestinationModel>().GetEnumerator();
			while (enumerator2.MoveNext())
			{
				DestinationModel current2 = enumerator2.Current;
				if (current2.demandLevelUpTime > _game.Scope.Get<ClockModel>().ExpansionTime)
				{
					current2.demandLevelUpTime = -Fix64.One;
				}
			}
			enumerator2 = _game.Simulation.GetModels<DestinationModel>().GetEnumerator();
			while (enumerator2.MoveNext())
			{
				DestinationModel current3 = enumerator2.Current;
				current3.unassignedDemand.Clear();
				current3.NextFrame.OvercrowdingTime = Fix64.Zero;
			}
			_gameUIScreen.UpgradeBar.RefreshAllAvailableUpgradeStacks();
			_gameUIScreen.UpgradeBar.HideAllNumberBubbles();
			break;
		}
		}
		_game.Scope.Get<AchievementCheckingProcess>().Reset();
		ReconfigurePermanenceVisibility();
		ResetLaneModelLaneSpeeds(_game.Simulation);
	}

	private void ResetLaneModelLaneSpeeds(ISimulation simulation)
	{
		ModelListEnumerator<LaneModel> enumerator = simulation.GetModels<LaneModel>().GetEnumerator();
		while (enumerator.MoveNext())
		{
			enumerator.Current.SetSpeedLimitScale(Fix64.One);
		}
	}

	private void ReconfigurePermanenceVisibility()
	{
		ViewClient viewClient = _game.Scope.Get<ViewClient>();
		foreach (RoundaboutView view in viewClient.GetViews<RoundaboutView>())
		{
			view.ReconfigurePermanenceVisibility();
		}
		foreach (TrafficLightView view2 in viewClient.GetViews<TrafficLightView>())
		{
			view2.ReconfigurePermanenceVisibility();
		}
		foreach (MotorwayView view3 in viewClient.GetViews<MotorwayView>())
		{
			view3.ReconfigurePermanenceVisibility();
		}
		foreach (TileView view4 in viewClient.GetViews<TileView>())
		{
			view4.ReconfigurePermanenceVisibility();
		}
	}

	public void SetRecentlyExitedCinematicMode(bool recentlyExitedCinematicMode)
	{
		_recentlyExitedCinematicMode = recentlyExitedCinematicMode;
	}

	public override void TransitionInTick()
	{
		base.TransitionInTick();
		float num = 1f / _constants.PercentageOfDurationToUseForInitialMovement;
		float gridAlpha = Mathf.Clamp01((TransitionInPercentage() - _constants.PercentageOfDurationToUseForInitialMovement) * -1f * num);
		_menuPlacementDefinition.SetGridAlpha(gridAlpha);
		if (_overrideTransitionInAnimation)
		{
			Vector3 cameraPositionForTransitionToGame = _constants.GetCameraPositionForTransitionToGame(_transitionDetails, TransitionInPercentage(), _newCity);
			_gameCamera.SetPosition(cameraPositionForTransitionToGame);
			if (TransitionInPercentage() < _constants.PercentageOfDurationToUseForInitialMovement)
			{
				_newCity.gameObject.SetActive(value: false);
			}
			else
			{
				_newCity.gameObject.SetActive(value: true);
			}
		}
		else
		{
			float num2 = Easings.CubicEaseInOut(TransitionInPercentage());
			Vector3 a = _transitionDetails.spline.Evaluate(num2);
			a = Vector3.Lerp(a, _game.Scope.Get<CameraView>().DesiredPosition, num2);
			_gameCamera.SetPosition(a);
		}
		if (TransitionInPercentage() > _percentageOfTransitionInToStartBluringForChallenges && !_game.PlayingBackSimJournal && _game.Simulation.GetModel<ActiveChallengesModel>().HasChallenges)
		{
			float num3 = 1f / (1f - _percentageOfTransitionInToStartBluringForChallenges);
			TransitionInPercentage();
			_ = _percentageOfTransitionInToStartBluringForChallenges;
		}
	}

	public override void TransitionOutTick()
	{
		base.TransitionOutTick();
		if (!_screenStack.IsScreenInStack(ScreenStack.MotorwaysScreen.InGame))
		{
			_menuPlacementDefinition.background.SetActive(value: true);
			_menuPlacementDefinition.grid.enabled = true;
			float num = 1f - _constants.PercentageOfDurationToUseForInitialMovement;
			float num2 = 1f / _constants.PercentageOfDurationToUseForInitialMovement;
			float gridAlpha = Mathf.Clamp01((TransitionOutPercentage() - num) * num2);
			_menuPlacementDefinition.SetGridAlpha(gridAlpha);
		}
		if (TransitionOutPercentage() > 1f - _constants.PercentageOfDurationToUseForInitialMovement && _isTransitioningOutToBeReleased && _newCity != null)
		{
			_game.Scope.Get<ViewClient>().SetAllGameObjectsEnabled(enabled: false);
			_newCity.gameObject.SetActive(value: false);
		}
	}

	public override void TransitionIn(ScreenStack.MotorwaysScreen outScreen)
	{
		StartupScreen activeScreen = _screenStack.GetActiveScreen<StartupScreen>();
		if (activeScreen != null)
		{
			activeScreen.StopMenuGameAudio();
		}
		_skipTransitions = _skipTransitions && outScreen != ScreenStack.MotorwaysScreen.GameOver;
		if (_startGameOnTransition)
		{
			_startGameOnTransition = false;
			if (_game == null)
			{
				_game = _appScope.Get<MotorwaysGame>();
			}
			_game.SetMapDefinition(_newMapDefinition);
			_game.Start(_newCity, _gameMode, _newMapChallenge);
			_game.SetPaused(isPaused: false, AnalyticsTypes.PauseReason.TransitionIn);
			_game.Tick(0f);
			_game.Tick(0f);
			_gameUIScreen = _game.Scope.Get<GameUIScreen>();
			backButton = _gameUIScreen.backButton;
			_cameraView = _game.Scope.Get<CameraView>();
			_game.OnGameStarted();
			if (_newMapChallenge != null)
			{
				ChallengeType? challengeType = null;
				switch (_newMapChallenge.type)
				{
				case MapChallenge.ChallengeType.Daily:
					challengeType = ChallengeType.Daily;
					break;
				case MapChallenge.ChallengeType.Weekly:
					challengeType = ChallengeType.Weekly;
					break;
				default:
					Diagnostics.FailAssert("Unknown challenge type for notifications. ({0})", _newMapChallenge.type);
					break;
				case MapChallenge.ChallengeType.Mystery:
				case MapChallenge.ChallengeType.City:
					break;
				}
				if (challengeType.HasValue)
				{
					_notificationEventSystem.RecordEvent(new PlayedChallenge
					{
						Type = challengeType.Value,
						TimeStart = _newMapChallenge.TimeStart
					});
				}
			}
			else
			{
				_notificationEventSystem.RecordEvent(new PlayedMap
				{
					Map = _newMapDefinition.CityNameEnum
				});
			}
			if (GetTransitionDuration() <= float.Epsilon)
			{
				_themeDatabase.SnapCurrentTransition();
			}
		}
		_softwareCapabilities.SetIsInMainMenuScreen(isInMainMenuScreen: false);
		_softwareCapabilities.SetIsInGame(isInGame: true);
		_softwareCapabilities.SetRichPresence(GetSteamRichPresenceTokens());
		base.TransitionIn(outScreen);
		_gameUIScreen.ToggleDebugLanguageSwitchButtons();
		_themeDatabase.SetCurrentMapDefinition(_newMapDefinition, GetTransitionDuration());
		(_runtimeAppCommandSource as RuntimeAppCommandSource)?.SetRewiredMode(2);
		_skipTransitions = _skipTransitions && outScreen != ScreenStack.MotorwaysScreen.GameOver;
		if (_newCity.CityTilemapMeshGenerator != null)
		{
			Material meshPreviewMaterials = _themeDatabase.bindings.materialCollection.materialBindings[28];
			_newCity.CityTilemapMeshGenerator.SetMeshPreviewMaterials(meshPreviewMaterials);
		}
	}

	private Dictionary<string, string> GetSteamRichPresenceTokens()
	{
		if (!(_softwareCapabilities is SteamSoftwareCapabilities))
		{
			return null;
		}
		return SteamSoftwareCapabilities.GetRichPresenceTokens(_newMapDefinition.cityName, ((_newMapChallenge != null) ? _newMapChallenge.type : MapChallenge.ChallengeType.None) switch
		{
			MapChallenge.ChallengeType.Daily => "#ModeDailyChallenge", 
			MapChallenge.ChallengeType.Weekly => "#ModeWeeklyChallenge", 
			_ => "#ModeCity", 
		});
	}

	public override void OnTransitionedIn()
	{
		base.OnTransitionedIn();
		if (!_playerPausedGame && !_startPaused && _recentlyExitedCinematicMode)
		{
			_gameUIScreen.OnPlayPressed();
		}
		else
		{
			_game.SetPaused(_playerPausedGame || _startPaused, AnalyticsTypes.PauseReason.TransitionIn);
		}
		_startPaused = false;
		_recentlyExitedCinematicMode = false;
		_gameUIScreen.OnTransitionedIn();
		_menuPlacementDefinition.background.SetActive(value: false);
		_menuPlacementDefinition.grid.enabled = false;
		StartupScreen activeScreen = _screenStack.GetActiveScreen<StartupScreen>();
		if (activeScreen != null)
		{
			activeScreen.StopSimulatingMenuGame();
		}
		_overrideTransitionInAnimation = false;
	}

	public override void TransitionOut(ScreenStack.MotorwaysScreen inScreen)
	{
		base.TransitionOut(inScreen);
		_playerPausedGame = _game.Simulation.IsPaused;
		_overrideTransitionInAnimation = false;
		_game.Scope.Get<GameUIScreen>().TransitionOut(inScreen);
		_softwareCapabilities.SetIsInGame(isInGame: false);
		if (inScreen == ScreenStack.MotorwaysScreen.MapSelect || inScreen == ScreenStack.MotorwaysScreen.ResumeGame || inScreen == ScreenStack.MotorwaysScreen.MainMenu || inScreen == ScreenStack.MotorwaysScreen.None)
		{
			_isTransitioningOutToBeReleased = true;
			StartupScreen activeScreen = _screenStack.GetActiveScreen<StartupScreen>();
			if (activeScreen != null)
			{
				activeScreen.StartSimulatingMenuGame();
				activeScreen.PlayMenuGameAudio();
			}
			_softwareCapabilities.SetRichPresence(null);
		}
		else
		{
			_isTransitioningOutToBeReleased = false;
		}
		(_runtimeAppCommandSource as RuntimeAppCommandSource)?.SetRewiredMode(0);
	}

	public override void OnLostFocus()
	{
		base.OnLostFocus();
		_playerActionController.TutorialBlockInputFlag = true;
	}

	public override void OnGainedFocus()
	{
		base.OnGainedFocus();
		_playerActionController.TutorialBlockInputFlag = false;
	}

	public override void ApplyTheme(ITheme newTheme)
	{
		base.ApplyTheme(newTheme);
		if ((bool)_gameUIScreen)
		{
			_gameUIScreen.ApplyTheme(newTheme);
		}
	}

	public override void ApplyBlendedTheme(ITheme oldTheme, ITheme newTheme, float progress)
	{
		base.ApplyBlendedTheme(oldTheme, newTheme, progress);
		if ((bool)_gameUIScreen)
		{
			_gameUIScreen.ApplyBlendedTheme(oldTheme, newTheme, progress);
		}
	}

	public override void Tick(float deltaTime)
	{
		base.Tick(deltaTime);
		if (!IsTransitioningIn() && !IsTransitioningOut())
		{
			if (!_screenStack.IsFading && !_game.PlayingBackSimJournal)
			{
				ActiveChallengesModel challengeModel = _game.Scope.Get<ActiveChallengesModel>();
				if (!_hasSeenChallenges && challengeModel.HasChallenges)
				{
					_screenStack.PushScreen(ScreenStack.MotorwaysScreen.ChallengeInfo, delegate(ChallengeInfoScreen screen)
					{
						MotorwaysGame game = _game;
						bool flag = game != null && game.Scope?.Get<ScoreModel>()?.Score == 0;
						screen.PrepareScreen(challengeModel.challengeType, challengeModel.challenges, challengeModel.timeStart, challengeModel.timeEnd, flag ? StringId.Begin : StringId.Continue, changeBlurWhenTransitioning: false, showBackButton: false, _game.Scope);
					});
					_hasSeenChallenges = true;
				}
				else if (!_hasSeenModeInfo && !challengeModel.HasChallenges && _screenStack.IsScreenVisible(ScreenStack.MotorwaysScreen.InGame))
				{
					if (_gameMode == GameMode.Endless && !_player.HasSeenNewContent("EndlessInfoPopupContentKey"))
					{
						ShowModeInfoPopup();
						_player.SetNewContentSeen("EndlessInfoPopupContentKey");
					}
					if (_gameMode == GameMode.Expert && !_player.HasSeenNewContent("ExpertInfoPopupContentKey"))
					{
						ShowModeInfoPopup();
						_player.SetNewContentSeen("ExpertInfoPopupContentKey");
					}
					if (_gameMode == GameMode.Creative && !_player.HasSeenNewContent("CreativeInfoPopupContentKey"))
					{
						ShowModeInfoPopup();
						_player.SetNewContentSeen("CreativeInfoPopupContentKey");
					}
				}
			}
			if (_game != null && ShouldTickGame())
			{
				_game.Tick(deltaTime);
			}
		}
		else if (_game != null)
		{
			_game.TickDuringTransition(deltaTime);
		}
	}

	private void ShowModeInfoPopup()
	{
		_startPaused = true;
		_gameUIScreen.OnPausePressed();
		popupStack.PushPopup<ModeInfoPopupInGame>().Initialize(_appScope, _gameMode, _gameUIScreen.OnPlayPressed);
		_hasSeenModeInfo = true;
	}

	private void OnApplicationPause(bool isPaused)
	{
		if (isPaused && _game != null)
		{
			_game.TrySave(GameJournalMotive.AppDeactivated);
			GameRules rules = _game.Scope.Get<City>().Rules;
			if (rules != null && rules.RecordsGameStatistics())
			{
				_game.RecordGameStatistics();
			}
		}
	}

	private void ReleaseGame()
	{
		if (Diagnostics.Verify(_game != null, "Trying to release a game when we don't have one!"))
		{
			_game.StopAudio();
			_game.ClearPathfinder();
			_game.Scope.ParentScope.Release(_game);
			_game = null;
			_playerPausedGame = false;
		}
	}

	public override void OnReleasedFromScope(IScope scope)
	{
		Log.Info("Disposing GameContainerScreen containing city {0}.", (_newCity != null) ? _newCity.name : "unknown");
		base.OnReleasedFromScope(scope);
		UnregisterThemeComponents();
		CleanupPreviousGame();
		StartupScreen activeScreen = _screenStack.GetActiveScreen<StartupScreen>();
		if (activeScreen != null)
		{
			activeScreen.StartSimulatingMenuGame();
		}
		_newMapDefinition = null;
		_newMapChallenge = null;
	}

	private void CleanupPreviousGame()
	{
		if (_newCity != null)
		{
			Object.Destroy(_newCity.gameObject);
			_newCity = null;
		}
		if (_game != null)
		{
			ReleaseGame();
		}
	}

	public void SetGameSuspended(bool suspendGame)
	{
		_gameSuspended = suspendGame;
	}

	private bool ShouldTickGame()
	{
		return !_gameSuspended;
	}

	public virtual Game GetActiveGame()
	{
		return _game;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
