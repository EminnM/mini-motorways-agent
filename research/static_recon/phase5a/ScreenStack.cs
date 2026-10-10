using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Factory;
using Motorways;
using Motorways.Screens;
using Motorways.Utility;
using Motorways.Views;
using Popups;
using Screens;
using TMPro;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.UI;
using Utils;

public class ScreenStack
{
	public enum MotorwaysScreen
	{
		None = -1,
		MainMenu,
		InGame,
		Pause,
		GameOver,
		Upgrade,
		OptionsMain,
		MapSelect,
		Credits,
		ResumeGame,
		Startup,
		Photo,
		ChallengeInfo,
		ProfileSelect,
		ProfileCreation,
		Movie,
		CinematicMode,
		OptionsPause,
		ArcadeIntroEndOfContent
	}

	private enum FadeStage
	{
		None,
		FadeToBlack,
		FadeFromBlack
	}

	public class MotorwaysScreenType
	{
		public MotorwaysScreen screenEnumType;

		public string assetBundle;

		public string prefabName;

		public IScreen screenInstance;

		public Type screenSystemType;

		public static MotorwaysScreenType ForScreenType<ScreenType>(MotorwaysScreen newScreenEnumType, string newAssetBundle, string newPrefabName, ScreenType newScreenInstance = null) where ScreenType : class, IScreen
		{
			MotorwaysScreenType motorwaysScreenType = new MotorwaysScreenType();
			motorwaysScreenType.screenEnumType = newScreenEnumType;
			motorwaysScreenType.assetBundle = newAssetBundle;
			motorwaysScreenType.prefabName = newPrefabName;
			motorwaysScreenType.screenSystemType = typeof(ScreenType);
			if (newScreenInstance != null && Diagnostics.Verify(motorwaysScreenType.screenSystemType.IsInstanceOfType(newScreenInstance), "We are trying to explicitly provide a screen instance for a type that it does not match!  Expected type {0}, but found type {1}.", typeof(ScreenType), motorwaysScreenType.GetType()))
			{
				motorwaysScreenType.screenInstance = newScreenInstance;
			}
			return motorwaysScreenType;
		}

		public IScreen GetScreenInstance(IScope appScope)
		{
			if (screenInstance == null)
			{
				object obj = appScope.Get(screenSystemType);
				if (Diagnostics.Verify(screenSystemType.IsAssignableFrom(obj.GetType()), "We tried to receive an instance of a screen, but the type doesn't match our expected type!  Expected type {0}, but found type {1}.", screenSystemType, obj.GetType()))
				{
					screenInstance = (IScreen)obj;
				}
				if (screenInstance == null && assetBundle != "" && prefabName != "")
				{
					GameObject gameObject = AssetBundleUtility.LoadPrefab(assetBundle, prefabName);
					if (Diagnostics.Verify(gameObject != null, "We were unable to load the screen prefab for screen {0} using asset bundle {1} and prefab name {2}.", screenEnumType.ToString(), assetBundle, prefabName))
					{
						obj = gameObject.GetComponentInChildren(screenSystemType);
						if (Diagnostics.Verify(obj != null, "We successfully loaded the prefab for screen {0}, but were unable to find the expected component of type {1} in the prefab.", screenEnumType.ToString(), screenSystemType.ToString()))
						{
							screenInstance = (IScreen)obj;
							Diagnostics.FailAssert("This is broken at the moment because we don't have a way to do dependency injection without allocating through the App.");
						}
					}
				}
			}
			return screenInstance;
		}
	}

	public static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("ScreenStack");

	private List<MotorwaysScreenType> _availableScreenTypes = new List<MotorwaysScreenType>
	{
		MotorwaysScreenType.ForScreenType<MainMenuScreen>(MotorwaysScreen.MainMenu, "core", "MainMenuScreen"),
		MotorwaysScreenType.ForScreenType<OptionsScreenMain>(MotorwaysScreen.OptionsMain, "core", "OptionsScreenMain"),
		MotorwaysScreenType.ForScreenType<OptionsScreenPause>(MotorwaysScreen.OptionsPause, "core", "OptionsScreenPause"),
		MotorwaysScreenType.ForScreenType<MapSelectScreen>(MotorwaysScreen.MapSelect, "core", "MapSelectScreen"),
		MotorwaysScreenType.ForScreenType<GameContainerScreen>(MotorwaysScreen.InGame, "core", "GameContainerScreen"),
		MotorwaysScreenType.ForScreenType<GameOverScreen>(MotorwaysScreen.GameOver, "core", "GameOverScreen"),
		MotorwaysScreenType.ForScreenType<GameUpgradeScreen>(MotorwaysScreen.Upgrade, "core", "GameUpgradeScreen"),
		MotorwaysScreenType.ForScreenType<PauseScreen>(MotorwaysScreen.Pause, "core", "PauseScreen"),
		MotorwaysScreenType.ForScreenType<ResumeGameScreen>(MotorwaysScreen.ResumeGame, "core", "ResumeGameScreen"),
		MotorwaysScreenType.ForScreenType<StartupScreen>(MotorwaysScreen.Startup, "core", "StartupScreen"),
		MotorwaysScreenType.ForScreenType<PhotoScreen>(MotorwaysScreen.Photo, "core", "PhotoScreen"),
		MotorwaysScreenType.ForScreenType<ChallengeInfoScreen>(MotorwaysScreen.ChallengeInfo, "core", "ChallengeInfoScreen"),
		MotorwaysScreenType.ForScreenType<ProfileSelectScreen>(MotorwaysScreen.ProfileSelect, "core", "ProfileSelectScreen"),
		MotorwaysScreenType.ForScreenType<ProfileCreationScreen>(MotorwaysScreen.ProfileCreation, "core", "ProfileCreationScreen"),
		MotorwaysScreenType.ForScreenType<MovieScreen>(MotorwaysScreen.Movie, "core", "MovieScreen"),
		MotorwaysScreenType.ForScreenType<CinematicModeScreen>(MotorwaysScreen.CinematicMode, "core", "CinematicModeScreen"),
		MotorwaysScreenType.ForScreenType<ArcadeIntroEndOfContentScreen>(MotorwaysScreen.ArcadeIntroEndOfContent, "core", "ArcadeIntroEndOfContent")
	};

	[Dependency]
	protected IScope _appScope;

	[Dependency]
	protected MenuPlacementDefinition _menuDefinition;

	[Dependency]
	protected GameCamera _camera;

	[Dependency]
	protected InputState _inputState;

	[Dependency]
	protected MotorwaysThemeDatabase _themeDatabase;

	[Dependency]
	protected ActivePlayer _player;

	public const float DemoIdleReturnDuration = 115f;

	private List<IScreen> _screenStack = new List<IScreen>();

	private List<IScreen> _activeScreens = new List<IScreen>();

	private List<IScreen> _visibleScreens = new List<IScreen>();

	private List<IScreen> _screensTransitioningOut = new List<IScreen>();

	private IScreen _screenTransitioningIn;

	private IScreen _pendingScreen;

	private bool _isPendingScreenAdditive;

	private const float _pendingScreenGraceDuration = 1.5f;

	private float _pendingScreenGraceTimer = -1f;

	private Image _fadeToBlackImage;

	private float _fadeTimer;

	private float _fadeDuration = 1f;

	private FadeStage _fadeStage;

	private bool _forceExitToMainMenu;

	private static readonly ProfilerMarker Profiler_Tick = new ProfilerMarker("ScreenStack.Tick");

	public bool AreAnyScreensTransitioning
	{
		get
		{
			if (_screenTransitioningIn == null)
			{
				return _screensTransitioningOut.Count > 0;
			}
			return true;
		}
	}

	public Canvas FadeToBlackCanvas { get; private set; }

	public bool IsFading => _fadeStage != FadeStage.None;

	public bool ExitingToMainMenu => _forceExitToMainMenu;

	public virtual ScreenType PushScreen<ScreenType>(MotorwaysScreen screenType, Action<ScreenType> prepAction, bool additive = false, IScope gameScope = null, bool blocksGameInput = true, IScreen overrideTransitionFrom = null) where ScreenType : class, IScreen
	{
		ScreenType val = CreateOrRetrieveScreenAndPrep(screenType, prepAction, gameScope, blocksGameInput);
		PushScreen(val, additive, overrideTransitionFrom);
		return val;
	}

	public virtual ScreenType PushScreen<ScreenType>(MotorwaysScreen screenType, bool additive = false, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		return PushScreen<ScreenType>(screenType, null, additive, gameScope, blocksGameInput);
	}

	public virtual IScreen PushScreen(MotorwaysScreen screenType, bool additive = false, IScope gameScope = null, bool blocksGameInput = true)
	{
		IScreen screen = CreateOrRetrieveScreen(screenType);
		InitializeInGameScreen(screen, gameScope, blocksGameInput);
		PushScreen(screen, additive);
		return screen;
	}

	protected virtual void PushScreen(IScreen newScreen, bool additive = false, IScreen overrideTransitionFrom = null)
	{
		if (newScreen.CanTransitionIn())
		{
			IScreen transitionOut = overrideTransitionFrom ?? ((_screenStack.Count > 0) ? _screenStack[_screenStack.Count - 1] : null);
			AddScreenToStack(newScreen);
			StartScreenTransitions(transitionOut, newScreen, additive);
			_pendingScreen = null;
		}
		else
		{
			_pendingScreen = newScreen;
			_isPendingScreenAdditive = additive;
			_pendingScreenGraceTimer = 1.5f;
		}
	}

	public virtual ScreenType ReplaceScreenOnTop<ScreenType>(MotorwaysScreen screenType, Action<ScreenType> prepAction, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		ScreenType val = CreateOrRetrieveScreenAndPrep(screenType, prepAction, gameScope, blocksGameInput);
		ReplaceScreenOnTop(val);
		return val;
	}

	public virtual ScreenType ReplaceScreenOnTop<ScreenType>(MotorwaysScreen screenType, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		return ReplaceScreenOnTop<ScreenType>(screenType, null, gameScope, blocksGameInput);
	}

	public virtual IScreen ReplaceScreenOnTop(MotorwaysScreen screenType, IScope gameScope = null, bool blocksGameInput = true)
	{
		IScreen screen = CreateOrRetrieveScreen(screenType);
		InitializeInGameScreen(screen, gameScope, blocksGameInput);
		ReplaceScreenOnTop(screen);
		return screen;
	}

	public virtual void ReplaceScreenOnTop(IScreen newScreen)
	{
		IScreen transitionOut = _screenStack[_screenStack.Count - 1];
		_screenStack.RemoveAt(_screenStack.Count - 1);
		AddScreenToStack(newScreen);
		StartScreenTransitions(transitionOut, newScreen);
	}

	public virtual ScreenType ReplaceScreens<ScreenType>(MotorwaysScreen screenType, Action<ScreenType> prepAction, Type includingMostRecentScreenOfType, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		ScreenType val = CreateOrRetrieveScreenAndPrep(screenType, prepAction, gameScope, blocksGameInput);
		ReplaceScreens(val, includingMostRecentScreenOfType);
		return val;
	}

	public virtual ScreenType ReplaceScreens<ScreenType>(MotorwaysScreen screenType, Type includingMostRecentScreenOfType, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		return ReplaceScreens<ScreenType>(screenType, null, includingMostRecentScreenOfType, gameScope, blocksGameInput);
	}

	public virtual IScreen ReplaceScreenOnTop(MotorwaysScreen screenType, Type includingMostRecentScreenOfType, IScope gameScope = null, bool blocksGameInput = true)
	{
		IScreen screen = CreateOrRetrieveScreen(screenType);
		InitializeInGameScreen(screen, gameScope, blocksGameInput);
		ReplaceScreens(screen, includingMostRecentScreenOfType);
		return screen;
	}

	public virtual void ReplaceScreens(IScreen newScreen, Type includingMostRecentScreenOfType)
	{
		IScreen transitionOut = _screenStack[_screenStack.Count - 1];
		IScreen screen = null;
		int num;
		for (num = _screenStack.Count - 1; num >= 0; num--)
		{
			if (includingMostRecentScreenOfType.IsAssignableFrom(_screenStack[num].GetType()))
			{
				screen = _screenStack[num];
				break;
			}
		}
		if (Diagnostics.Verify(screen != null, "We were unable to find a screen of type {0} in the stack!  Aborting the ReplaceScreens().", includingMostRecentScreenOfType.ToString()))
		{
			for (int i = num; i < _screenStack.Count; i++)
			{
				StartScreenTransitions(_screenStack[i], null);
			}
			_screenStack.RemoveRange(num, _screenStack.Count - num);
			AddScreenToStack(newScreen);
			StartScreenTransitions(transitionOut, newScreen);
		}
	}

	public virtual void PopOneScreen()
	{
		if (Diagnostics.Verify(_screenStack.Count > 1, "Trying to pop back a screen when we only have {0} screens.", _screenStack.Count))
		{
			IScreen screen = _screenStack[_screenStack.Count - 1];
			if (screen.CanPopScreen())
			{
				_screenStack.RemoveAt(_screenStack.Count - 1);
				_screenTransitioningIn = _screenStack[_screenStack.Count - 1];
				StartScreenTransitions(screen, _screenTransitioningIn);
			}
			else
			{
				Debug.Log("Cant pop screen: " + screen);
			}
		}
	}

	public virtual void PopToScreenOfType(MotorwaysScreen screenType, bool inclusive = false)
	{
		PopToScreenOfType(GetScreenTypeForEnum(screenType).screenSystemType, inclusive);
	}

	public virtual void PopToScreenOfType(Type screenType, bool inclusive = false)
	{
		Log.Info("Popping {0} screen {1}.", inclusive ? "past" : "to", screenType);
		IScreen screen = null;
		IScreen screen2 = null;
		int num = -1;
		for (int num2 = _screenStack.Count - 1; num2 >= 0; num2--)
		{
			Log.Info("Checking screen {0} ...", _screenStack[num2].GetType());
			if (screen == null && screenType.IsAssignableFrom(_screenStack[num2].GetType()))
			{
				screen = _screenStack[num2];
				if (!inclusive)
				{
					screen2 = screen;
					num = num2 + 1;
					break;
				}
			}
			else if (screen != null)
			{
				screen2 = _screenStack[num2];
				num = num2 + 1;
				break;
			}
		}
		if (!Diagnostics.Verify(screen != null, "We were unable to find a screen of type {0} in the stack! Aborting the PopBackToScreenOfType().", screenType.ToString()) || !Diagnostics.Verify(num >= 0, "We didn't find a final screen to arrive at out of {0} screens.", _screenStack.Count))
		{
			return;
		}
		for (int i = num; i < _screenStack.Count - 1; i++)
		{
			if (Diagnostics.Verify(i < _screenStack.Count - 1 && i >= 0, "Screen index out of bounds at {0} of {1}", i, _screenStack.Count))
			{
				StartScreenTransitions(_screenStack[i], null);
			}
		}
		try
		{
			StartScreenTransitions(_screenStack[_screenStack.Count - 1], screen2);
		}
		catch (Exception ex)
		{
			Diagnostics.FailAssert("A error occured when transitioning to {0} raised exception: {1}", screen2?.GetType(), ex.ToString());
		}
		Log.Info("Removing {0} screens", _screenStack.Count - num);
		_screenStack.RemoveRange(num, _screenStack.Count - num);
	}

	public virtual IScreen CreateOrRetrieveScreen(MotorwaysScreen screenType)
	{
		return GetScreenTypeForEnum(screenType).GetScreenInstance(_appScope);
	}

	public virtual ScreenType CreateOrRetrieveScreen<ScreenType>(MotorwaysScreen screenType) where ScreenType : class, IScreen
	{
		IScreen screenInstance = GetScreenTypeForEnum(screenType).GetScreenInstance(_appScope);
		if (Diagnostics.Verify(screenInstance != null, "We were unable to get a screen instance for screen type {0}.", screenType.ToString()) && Diagnostics.Verify(typeof(ScreenType).IsAssignableFrom(screenInstance.GetType()), "We got a screen for type {0}, but the type returned {1} doesn't match the requested system type {2}.", screenType.ToString(), screenInstance.GetType().ToString(), typeof(ScreenType).ToString()))
		{
			ScreenType val = (ScreenType)screenInstance;
			if (Diagnostics.Verify(val != null, "We failed to cast the properly generated {0} screen to the system type {1}.", screenType.ToString(), typeof(ScreenType)))
			{
				return val;
			}
		}
		return null;
	}

	public virtual ScreenType CreateOrRetrieveScreenAndPrep<ScreenType>(MotorwaysScreen screenType, Action<ScreenType> prepareAction, IScope gameScope = null, bool blocksGameInput = true) where ScreenType : class, IScreen
	{
		ScreenType val = CreateOrRetrieveScreen<ScreenType>(screenType);
		if (val != null)
		{
			InitializeInGameScreen(val, gameScope, blocksGameInput);
			val.Enable(shouldBeVisible: true);
			prepareAction?.Invoke(val);
		}
		return val;
	}

	public virtual ScreenType GetActiveScreen<ScreenType>() where ScreenType : IScreen
	{
		return (ScreenType)GetActiveScreen(typeof(ScreenType));
	}

	public virtual IScreen GetActiveScreen(MotorwaysScreen screenType)
	{
		return GetActiveScreen(GetScreenTypeForEnum(screenType).screenSystemType);
	}

	public virtual IScreen GetActiveScreen(Type systemType)
	{
		for (int i = 0; i < _activeScreens.Count; i++)
		{
			if (systemType.IsAssignableFrom(_activeScreens[i].GetType()))
			{
				return _activeScreens[i];
			}
		}
		return null;
	}

	public IEnumerable<IScreen> GetActiveScreens()
	{
		return _activeScreens;
	}

	public MotorwaysScreen GetScreenTypeBelowScreenType(MotorwaysScreen screenType)
	{
		for (int i = 1; i < _activeScreens.Count; i++)
		{
			if (_activeScreens[i].GetType() == GetScreenTypeForEnum(screenType).screenSystemType)
			{
				return GetScreenTypeForSystemType(_activeScreens[i - 1].GetType()).screenEnumType;
			}
		}
		return MotorwaysScreen.None;
	}

	public virtual bool IsScreenInStack<ScreenType>() where ScreenType : IScreen
	{
		return IsScreenInStack(typeof(ScreenType));
	}

	public virtual bool IsScreenInStack(Type systemType)
	{
		for (int i = 0; i < _screenStack.Count; i++)
		{
			if (systemType.IsAssignableFrom(_screenStack[i].GetType()))
			{
				return true;
			}
		}
		return false;
	}

	public virtual bool IsScreenInStack(MotorwaysScreen screenType)
	{
		return IsScreenInStack(GetScreenTypeForEnum(screenType).screenSystemType);
	}

	public virtual bool IsScreenVisible<ScreenType>() where ScreenType : IScreen
	{
		return IsScreenVisible(typeof(ScreenType));
	}

	public virtual bool IsScreenVisible(Type systemType)
	{
		for (int i = 0; i < _visibleScreens.Count; i++)
		{
			if (systemType.IsAssignableFrom(_visibleScreens[i].GetType()))
			{
				return true;
			}
		}
		return false;
	}

	public virtual bool IsScreenVisible(MotorwaysScreen screenType)
	{
		return IsScreenVisible(GetScreenTypeForEnum(screenType).screenSystemType);
	}

	public virtual bool IsScreenActive<ScreenType>() where ScreenType : IScreen
	{
		return IsScreenActive(typeof(ScreenType));
	}

	public virtual bool IsScreenActive(Type systemType)
	{
		for (int i = 0; i < _activeScreens.Count; i++)
		{
			if (systemType.IsInstanceOfType(_activeScreens[i]))
			{
				return true;
			}
		}
		return false;
	}

	public virtual bool IsScreenActive(MotorwaysScreen screenType)
	{
		return IsScreenActive(GetScreenTypeForEnum(screenType).screenSystemType);
	}

	public virtual bool IsScreenPending<ScreenType>() where ScreenType : IScreen
	{
		if (_pendingScreen != null)
		{
			return _pendingScreen.GetType() == typeof(ScreenType);
		}
		return false;
	}

	public bool HasPendingScreen()
	{
		return _pendingScreen != null;
	}

	public virtual bool IsInGame()
	{
		return GetScreenTypeForEnum(MotorwaysScreen.InGame).screenSystemType.IsAssignableFrom(_screenStack[_screenStack.Count - 1].GetType());
	}

	public void FadeNextTransition(float duration)
	{
		_fadeDuration = duration;
		_fadeTimer = duration;
		_fadeStage = FadeStage.FadeToBlack;
	}

	public virtual Game GetGameIfInGame()
	{
		GameContainerScreen activeScreen = GetActiveScreen<GameContainerScreen>();
		if (activeScreen != null)
		{
			return activeScreen.GetActiveGame();
		}
		return null;
	}

	public virtual MotorwaysScreen CurrentVisibleScreenType()
	{
		return GetScreenTypeForSystemType(_visibleScreens[_visibleScreens.Count - 1].GetType()).screenEnumType;
	}

	public bool HasVisibleScreens()
	{
		return _visibleScreens.Count > 0;
	}

	public virtual IScreen GetTopVisibleScreen()
	{
		if (!Diagnostics.Verify(_visibleScreens.Count >= 1, "Trying to get a visible screen when we don't have one"))
		{
			return null;
		}
		return _visibleScreens[_visibleScreens.Count - 1];
	}

	public MotorwaysScreen GetTopActiveScreenType()
	{
		if (GetScreenTypeForSystemType(_activeScreens[_activeScreens.Count - 1].GetType()) == null)
		{
			return MotorwaysScreen.None;
		}
		return GetScreenTypeForSystemType(_activeScreens[_activeScreens.Count - 1].GetType()).screenEnumType;
	}

	private bool IsInGameScreen(MotorwaysScreen screen)
	{
		if (_menuDefinition.IsInGameScreen(screen))
		{
			return true;
		}
		if (GetGameIfInGame() != null && screen == MotorwaysScreen.ChallengeInfo)
		{
			return true;
		}
		return false;
	}

	public Vector3 GetPositionFor(MotorwaysScreen screen)
	{
		if (IsInGameScreen(screen))
		{
			if (screen == MotorwaysScreen.GameOver)
			{
				GameOverScreen activeScreen = GetActiveScreen<GameOverScreen>();
				if (activeScreen != null)
				{
					return activeScreen.focusPoint;
				}
			}
			Game gameIfInGame = GetGameIfInGame();
			if (Diagnostics.Verify(gameIfInGame != null, "Game can't be null by the time we're transitioning to the game!"))
			{
				CameraView cameraView = gameIfInGame.Scope.Get<CameraView>();
				if (screen == MotorwaysScreen.InGame)
				{
					return cameraView.DesiredPosition;
				}
				return cameraView.CurrentUnfocusedPosition;
			}
		}
		if (screen == MotorwaysScreen.MapSelect || screen == MotorwaysScreen.ResumeGame || screen == MotorwaysScreen.ProfileSelect)
		{
			ScrollingButtonScreen scrollingButtonScreen = null;
			switch (screen)
			{
			case MotorwaysScreen.MapSelect:
				scrollingButtonScreen = GetActiveScreen<MapSelectScreen>();
				break;
			case MotorwaysScreen.ResumeGame:
				scrollingButtonScreen = GetActiveScreen<ResumeGameScreen>();
				break;
			case MotorwaysScreen.ProfileSelect:
				scrollingButtonScreen = GetActiveScreen<ProfileSelectScreen>();
				break;
			}
			if (scrollingButtonScreen != null && scrollingButtonScreen.HasValidCameraPosition())
			{
				return scrollingButtonScreen.GetCameraPosition();
			}
		}
		return _menuDefinition.GetPositionFor(screen);
	}

	public Quaternion GetRotationFor(MotorwaysScreen screen)
	{
		return _menuDefinition.GetRotationFor(screen);
	}

	public float GetZoomFor(MotorwaysScreen screen)
	{
		if (screen == MotorwaysScreen.GameOver)
		{
			return _menuDefinition.GetZoomFor(screen);
		}
		if (_menuDefinition.IsInGameScreen(screen))
		{
			Game gameIfInGame = GetGameIfInGame();
			if (Diagnostics.Verify(gameIfInGame != null, "Game can't be null by the time we're transitioning to the game!"))
			{
				CameraView cameraView = gameIfInGame.Scope.Get<CameraView>();
				cameraView.UpdateMaxZoom();
				return cameraView.MaxZoom;
			}
		}
		return _menuDefinition.GetZoomFor(screen);
	}

	public ScreenTransition GetTransitionDetailsFrom(MotorwaysScreen origin, MotorwaysScreen destination)
	{
		if (Diagnostics.Verify(_menuDefinition != null, "_menuDefinition is null! : ScreenStack.GetTransitionDetailsFrom()"))
		{
			NodeConnection connectionFrom = _menuDefinition.GetConnectionFrom(origin, destination);
			Quaternion rotationFor = GetRotationFor(connectionFrom.startNode.screen);
			Vector3 positionFor = GetPositionFor(connectionFrom.startNode.screen);
			Vector3 entryHandle = connectionFrom.entryHandle;
			Vector3 exitHandle = connectionFrom.exitHandle;
			Vector3 positionFor2 = GetPositionFor(connectionFrom.endNode.screen);
			Quaternion rotationFor2 = GetRotationFor(connectionFrom.endNode.screen);
			return new ScreenTransition
			{
				spline = new Spline.BezierSplineWithRotation(positionFor, positionFor + entryHandle, exitHandle + positionFor2, positionFor2, rotationFor, rotationFor2),
				duration = connectionFrom.duration,
				cameraControl = connectionFrom.cameraControl
			};
		}
		return null;
	}

	public virtual void DONTCALL_RegisterTestScreenType<ScreenType>(MotorwaysScreen newScreenEnumType, string newAssetBundle, string newPrefabName, ScreenType newScreenInstance = null) where ScreenType : class, IScreen
	{
		if (Diagnostics.Verify(GetScreenTypeForSystemType(typeof(ScreenType)) == null, "We shouldn't have a screen of type {0} already registered!", typeof(ScreenType).ToString()))
		{
			_availableScreenTypes.Add(MotorwaysScreenType.ForScreenType(newScreenEnumType, newAssetBundle, newPrefabName, newScreenInstance));
		}
	}

	public virtual void Start()
	{
		FadeToBlackCanvas = new GameObject("FadeToBlack").AddComponent<Canvas>();
		FadeToBlackCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
		_camera.AttachCameraToCanvas(FadeToBlackCanvas, CameraLayer.Default);
		FadeToBlackCanvas.sortingLayerName = "UI";
		FadeToBlackCanvas.sortingOrder = 10;
		if (FeatureToggle.IsFeatureEnabled(Feature.BetaWatermark))
		{
			TextMeshProUGUI textMeshProUGUI = new GameObject("Text").AddComponent<TextMeshProUGUI>();
			textMeshProUGUI.SetText($"CONFIDENTIAL\nMini Motorways {Version.Name} ({Version.Timestamp})");
			textMeshProUGUI.color = Color.grey;
			textMeshProUGUI.gameObject.transform.SetParent(FadeToBlackCanvas.gameObject.transform, worldPositionStays: false);
			RectTransform component = textMeshProUGUI.GetComponent<RectTransform>();
			component.pivot = new Vector2(0f, 0.5f);
			component.sizeDelta = new Vector2(1000f, 100f);
			component.anchorMax = Vector2.zero;
			component.anchorMin = Vector2.zero;
			component.anchoredPosition = new Vector2(300f, 150f);
			Canvas canvas = UnityEngine.Object.Instantiate(FadeToBlackCanvas);
			canvas.gameObject.name = "BetaTextOverlayCanvas";
			_camera.AttachCameraToCanvas(canvas, CameraLayer.Overlay);
			canvas.gameObject.layer = _camera.OverlayLayerIndex;
		}
		_fadeToBlackImage = FadeToBlackCanvas.gameObject.AddComponent<Image>();
		_fadeToBlackImage.color = Color.clear;
		PushScreen(_appScope.Get<IInitialGameScreen>());
	}

	public virtual void Tick(float deltaTime)
	{
		if (_pendingScreen != null && _pendingScreen.CanTransitionIn())
		{
			_pendingScreenGraceTimer -= deltaTime;
			if (_pendingScreenGraceTimer <= 0f)
			{
				PushScreen(_pendingScreen, _isPendingScreenAdditive);
				_pendingScreen = null;
			}
		}
		else if (_pendingScreen != null)
		{
			_pendingScreenGraceTimer = 1.5f;
		}
		if (_fadeTimer > 0f)
		{
			_fadeTimer -= deltaTime;
			if (_fadeStage == FadeStage.FadeToBlack)
			{
				_fadeToBlackImage.color = Color.Lerp(Color.clear, Color.black, 1f - _fadeTimer / _fadeDuration);
			}
			else if (_fadeStage == FadeStage.FadeFromBlack)
			{
				_fadeToBlackImage.color = Color.Lerp(Color.black, Color.clear, 1f - _fadeTimer / _fadeDuration);
				if (_fadeTimer < 0f)
				{
					_fadeStage = FadeStage.None;
				}
			}
		}
		else
		{
			for (int i = 0; i < _activeScreens.Count; i++)
			{
				_activeScreens[i].Tick(deltaTime);
			}
		}
		if (_screensTransitioningOut.Count > 0)
		{
			for (int num = _screensTransitioningOut.Count - 1; num >= 0; num--)
			{
				IScreen screen = _screensTransitioningOut[num];
				if (!screen.IsTransitioningOut())
				{
					Log.Info("Screen {0} has transitioned out.", screen);
					screen.OnTransitionedOut();
					if (!_screenStack.Contains(screen))
					{
						_activeScreens.Remove(screen);
						RemoveScreenInstanceOfType(GetScreenEnumForSystemType(screen.GetType()));
					}
					_screensTransitioningOut.Remove(screen);
				}
			}
		}
		if (_screenTransitioningIn != null && !_screenTransitioningIn.IsTransitioningIn())
		{
			Log.Info("Screen {0} has transitioned in.", _screenTransitioningIn);
			if (_fadeTimer <= 0f && _fadeStage == FadeStage.FadeToBlack)
			{
				_fadeStage = FadeStage.FadeFromBlack;
				_fadeTimer = _fadeDuration;
			}
			if (!_visibleScreens.Contains(_screenTransitioningIn))
			{
				_visibleScreens.Add(_screenTransitioningIn);
			}
			_screenTransitioningIn.OnTransitionedIn();
			_screenTransitioningIn = null;
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.AppleStoreDemo) && _screenTransitioningIn == null && _screensTransitioningOut.Count == 0)
		{
			MotorwaysScreen topActiveScreenType = GetTopActiveScreenType();
			if (topActiveScreenType != MotorwaysScreen.None && topActiveScreenType != MotorwaysScreen.MainMenu && _screenTransitioningIn == null && (_visibleScreens.Count == 0 || CurrentVisibleScreenType() != MotorwaysScreen.MainMenu) && (Time.time - _inputState.LastInputTimestamp > 115f || _forceExitToMainMenu))
			{
				_forceExitToMainMenu = false;
				_appScope.Get<PopupStack>().PushPopup<AppleDemoCardPopup>(0f, ignoreScreen: true).Initialise(showFrontCard: true);
				_inputState.BlockAllInput = true;
				if (topActiveScreenType == MotorwaysScreen.Upgrade || topActiveScreenType == MotorwaysScreen.Photo || topActiveScreenType == MotorwaysScreen.CinematicMode)
				{
					PopOneScreen();
					_forceExitToMainMenu = true;
				}
				else
				{
					_themeDatabase.SetCurrentMapDefinition(GetActiveScreen<StartupScreen>().mapDefinition, 1f);
					GameContainerScreen activeScreen = GetActiveScreen<GameContainerScreen>();
					if (activeScreen != null)
					{
						Game activeGame = activeScreen.GetActiveGame();
						activeGame.StopAudio();
						if (topActiveScreenType != MotorwaysScreen.GameOver)
						{
							activeGame.Scope.Get<GameUIScreen>().SetUIVisible(visible: false, instantly: false, forceHide: true);
							activeGame.OnGameEnd(GameEndReason.Exit);
						}
					}
					PopToScreenOfType(MotorwaysScreen.MainMenu);
					_inputState.BlockAllInput = false;
				}
				_player.MotorwaysUserProfile.ClearCityStatistics();
			}
		}
		if (!_forceExitToMainMenu || FeatureToggle.IsFeatureEnabled(Feature.AppleStoreDemo) || AreAnyScreensTransitioning)
		{
			return;
		}
		MotorwaysScreen topActiveScreenType2 = GetTopActiveScreenType();
		if (!_menuDefinition.TransitionExists(topActiveScreenType2, MotorwaysScreen.MainMenu))
		{
			PopOneScreen();
			return;
		}
		GameContainerScreen activeScreen2 = GetActiveScreen<GameContainerScreen>();
		if (activeScreen2 != null)
		{
			Game activeGame2 = activeScreen2.GetActiveGame();
			activeGame2.StopAudio();
			if (topActiveScreenType2 != MotorwaysScreen.GameOver)
			{
				activeGame2.Scope.Get<GameUIScreen>().SetUIVisible(visible: false, instantly: false, forceHide: true);
				activeGame2.OnGameEnd(GameEndReason.Exit);
			}
		}
		_themeDatabase.SetCurrentMapDefinition(GetActiveScreen<StartupScreen>().mapDefinition, 1f);
		PopToScreenOfType(MotorwaysScreen.MainMenu);
	}

	public MotorwaysScreen GetScreenEnumForSystemType(Type screenType)
	{
		return GetScreenTypeForSystemType(screenType)?.screenEnumType ?? MotorwaysScreen.None;
	}

	public async Task ExitToMainMenu()
	{
		_forceExitToMainMenu = true;
		_inputState.BlockAllInput = true;
		CameraBlurEffect cameraBlurEffect = new CameraBlurEffect(this, 1f, 1f);
		_camera.AddBlurEffect(cameraBlurEffect);
		while (!IsInMainMenu())
		{
			await Task.Delay(1);
			if (!(_screenTransitioningIn is MainMenuScreen))
			{
				continue;
			}
			while (cameraBlurEffect.blurRatio > 0f)
			{
				cameraBlurEffect.blurRatio -= Time.deltaTime * 0.5f;
				if (IsInMainMenu())
				{
					break;
				}
				await Task.Delay(1);
			}
			cameraBlurEffect.blurRatio = 0f;
		}
		_camera.RemoveBlurEffect(cameraBlurEffect);
		_inputState.BlockAllInput = false;
		_forceExitToMainMenu = false;
	}

	private bool IsInMainMenu()
	{
		if (HasVisibleScreens() && !AreAnyScreensTransitioning)
		{
			return GetTopVisibleScreen() is MainMenuScreen;
		}
		return false;
	}

	public void OnApplicationPaused()
	{
		if (!FeatureToggle.IsFeatureEnabled(Feature.AppleStoreDemo))
		{
			return;
		}
		MotorwaysScreen topActiveScreenType = GetTopActiveScreenType();
		if (topActiveScreenType == MotorwaysScreen.None || topActiveScreenType == MotorwaysScreen.MainMenu || topActiveScreenType == MotorwaysScreen.Startup || _screenTransitioningIn != null || (_visibleScreens.Count != 0 && CurrentVisibleScreenType() == MotorwaysScreen.MainMenu))
		{
			return;
		}
		_forceExitToMainMenu = false;
		_inputState.BlockAllInput = true;
		if (topActiveScreenType == MotorwaysScreen.Upgrade || topActiveScreenType == MotorwaysScreen.Photo || topActiveScreenType == MotorwaysScreen.CinematicMode)
		{
			PopOneScreen();
			_forceExitToMainMenu = true;
		}
		else
		{
			_appScope.Get<PopupStack>().PushPopup<AppleDemoCardPopup>(0f, ignoreScreen: true).Initialise(showFrontCard: true);
			_themeDatabase.SetCurrentMapDefinition(GetActiveScreen<StartupScreen>().mapDefinition, 1f);
			GameContainerScreen activeScreen = GetActiveScreen<GameContainerScreen>();
			if (activeScreen != null)
			{
				Game activeGame = activeScreen.GetActiveGame();
				activeGame.StopAudio();
				if (topActiveScreenType != MotorwaysScreen.GameOver)
				{
					activeGame.Scope.Get<GameUIScreen>().SetUIVisible(visible: false, instantly: false, forceHide: true);
					activeGame.OnGameEnd(GameEndReason.Exit);
				}
			}
			PopToScreenOfType(MotorwaysScreen.MainMenu);
			_inputState.BlockAllInput = false;
		}
		_player.MotorwaysUserProfile.ClearCityStatistics();
	}

	protected virtual void AddScreenToStack(IScreen newScreen)
	{
		_screenStack.Add(newScreen);
		_activeScreens.Add(newScreen);
		newScreen.Enable(shouldBeVisible: true);
	}

	protected virtual void StartScreenTransitions(IScreen transitionOut, IScreen transitionIn, bool additive = false)
	{
		_screenTransitioningIn = transitionIn;
		if (_screenTransitioningIn != null)
		{
			MotorwaysScreen outScreen = MotorwaysScreen.None;
			if (transitionOut != null)
			{
				MotorwaysScreenType screenTypeForSystemType = GetScreenTypeForSystemType(transitionOut.GetType());
				if (screenTypeForSystemType != null)
				{
					outScreen = screenTypeForSystemType.screenEnumType;
				}
			}
			_screenTransitioningIn.TransitionIn(outScreen);
		}
		if (transitionOut == null)
		{
			return;
		}
		if (!_screensTransitioningOut.Contains(transitionOut))
		{
			_screensTransitioningOut.Add(transitionOut);
		}
		MotorwaysScreen inScreen = MotorwaysScreen.None;
		if (transitionIn != null)
		{
			MotorwaysScreenType screenTypeForSystemType2 = GetScreenTypeForSystemType(transitionIn.GetType());
			if (screenTypeForSystemType2 != null)
			{
				inScreen = screenTypeForSystemType2.screenEnumType;
			}
		}
		transitionOut.TransitionOut(inScreen);
		if (!additive && _visibleScreens.Contains(transitionOut))
		{
			_visibleScreens.Remove(transitionOut);
		}
	}

	protected virtual MotorwaysScreenType GetScreenTypeForEnum(MotorwaysScreen screenType)
	{
		for (int i = 0; i < _availableScreenTypes.Count; i++)
		{
			if (_availableScreenTypes[i].screenEnumType == screenType)
			{
				return _availableScreenTypes[i];
			}
		}
		return null;
	}

	protected virtual void RemoveScreenInstanceOfType(MotorwaysScreen screenType)
	{
		for (int i = 0; i < _availableScreenTypes.Count; i++)
		{
			if (_availableScreenTypes[i].screenEnumType == screenType)
			{
				if (_availableScreenTypes[i].screenInstance != null)
				{
					_appScope.Release(_availableScreenTypes[i].screenInstance);
				}
				_availableScreenTypes[i].screenInstance = null;
				break;
			}
		}
	}

	protected virtual MotorwaysScreenType GetScreenTypeForSystemType(Type screenType)
	{
		for (int i = 0; i < _availableScreenTypes.Count; i++)
		{
			if (_availableScreenTypes[i].screenSystemType == screenType)
			{
				return _availableScreenTypes[i];
			}
		}
		return null;
	}

	protected virtual bool InitializeInGameScreen(IScreen screenInst, IScope withGameScope, bool blocksGameInput)
	{
		if (typeof(InGameScalingScreen).IsAssignableFrom(screenInst.GetType()) && Diagnostics.Verify(withGameScope != null, "We are attempting to init a {0} screen which requires a game Scope to be initialized, but one was not provided!", screenInst.GetType()))
		{
			InGameScalingScreen inGameScalingScreen = screenInst as InGameScalingScreen;
			if (inGameScalingScreen != null)
			{
				inGameScalingScreen.InitScreen(withGameScope, blocksGameInput);
				return true;
			}
		}
		return false;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
