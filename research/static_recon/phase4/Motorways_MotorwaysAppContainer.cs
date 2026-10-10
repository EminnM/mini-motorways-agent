using Analytics;
using Client;
using Factory;
using Factory.Allocators;
using Factory.Pools;
using GameEvent;
using Motorways.Actions;
using Motorways.Commands;
using Motorways.Leaderboards;
using Motorways.Models;
using Motorways.Processes;
using Motorways.Screens;
using Motorways.UI;
using Motorways.UI.NewContentIndicators;
using Motorways.Utility;
using Motorways.Views;
using Motorways.Views.Boats;
using Motorways.Views.MeshGeneration;
using Motorways.Views.Trains;
using Popups;
using Server;
using UnityEngine;

namespace Motorways;

public class MotorwaysAppContainer : AppContainer
{
	protected override void RegisterSerializers()
	{
		SerializerLibrary.RegisterSerializer<TileDirectionBitfield>(new TileDirectionBitfield.Serializer());
		SerializerLibrary.RegisterSerializer<CornerAdjacencyReference>(new CornerAdjacencyReference.Serializer());
		SerializerLibrary.RegisterSerializer<UpgradePackageDefinition>(new UpgradePackageDefinition.Serializer());
		SerializerLibrary.RegisterSerializer<RoadTileConnection>(new RoadTileConnection.Serializer());
		SerializerLibrary.RegisterSerializer<RoadTileNode>(new RoadTileNode.Serializer());
		SerializerLibrary.RegisterSerializer<RailTileConnection>(new RailTileConnection.Serializer());
		SerializerLibrary.RegisterSerializer<BoatPathTileConnection>(new BoatPathTileConnection.Serializer());
		SerializerLibrary.RegisterSerializer<PlannedBuilding>(new PlannedBuilding.Serializer());
		SerializerLibrary.RegisterSerializer<Spline.BezierSpline>(new Spline.BezierSpline.Serializer());
		SerializerLibrary.RegisterSerializer<Spline.BezierSplineFixed>(new Spline.BezierSplineFixed.Serializer());
		SerializerLibrary.RegisterSerializer<ChallengeData>(new ChallengeData.Serializer());
		SerializerLibrary.RegisterSerializer<AdjacentTileConnection>(new AdjacentTileConnection.Serializer());
	}

	protected override Assembler CreateAppAssembler()
	{
		Assembler assembler = base.CreateAppAssembler();
		assembler.Register<GameEventBroadcaster>().Allocator(new SingletonAllocator<GameEventBroadcaster>(new GameEventBroadcaster())).Binding(Binding.Scope);
		assembler.Register<PseudorandomGenerator>().Allocator(new ObjectPool<PseudorandomGenerator>
		{
			InitialSize = 4
		});
		MotorwaysThemeDatabaseBindings themeBindings = AssetBundleUtility.LoadAsset<MotorwaysThemeDatabaseBindings>("core", "ThemeDatabaseBindings");
		assembler.Register<IThemeDatabase, MotorwaysThemeDatabase>().Allocator(new SingletonAllocator<MotorwaysThemeDatabase>(new MotorwaysThemeDatabase(themeBindings))).Binding(Binding.Scope);
		SimulationConstantsData instance = AssetBundleUtility.LoadAsset<SimulationConstantsData>("core", "SimulationConstantsData");
		assembler.Register<SimulationConstantsData, SimulationConstantsData>().Allocator(new SingletonAllocator<SimulationConstantsData>(instance)).Binding(Binding.Scope);
		VisualConstantsData instance2 = AssetBundleUtility.LoadAsset<VisualConstantsData>("core", "VisualConstantsData");
		assembler.Register<VisualConstantsData, VisualConstantsData>().Allocator(new SingletonAllocator<VisualConstantsData>(instance2)).Binding(Binding.Scope);
		PermanenceTextureMappingDatabase instance3 = AssetBundleUtility.LoadAsset<PermanenceTextureMappingDatabase>("core", "PermanenceTextureMappingDatabase");
		assembler.Register<PermanenceTextureMappingDatabase, PermanenceTextureMappingDatabase>().Allocator(new SingletonAllocator<PermanenceTextureMappingDatabase>(instance3)).Binding(Binding.Scope);
		MotorwayVisualParameters instance4 = AssetBundleUtility.LoadAsset<MotorwayVisualParameters>("core", "MotorwayVisualParameters");
		assembler.Register<MotorwayVisualParameters, MotorwayVisualParameters>().Allocator(new SingletonAllocator<MotorwayVisualParameters>(instance4)).Binding(Binding.Scope);
		RoadTileConstantsData instance5 = AssetBundleUtility.LoadAsset<RoadTileConstantsData>("core", "RoadTileConstantsData");
		assembler.Register<RoadTileConstantsData>().Allocator(new SingletonAllocator<RoadTileConstantsData>(instance5)).Binding(Binding.Scope);
		TutorialConstantsData instance6 = AssetBundleUtility.LoadAsset<TutorialConstantsData>("core", "TutorialConstantsData");
		assembler.Register<TutorialConstantsData, TutorialConstantsData>().Allocator(new SingletonAllocator<TutorialConstantsData>(instance6)).Binding(Binding.Scope);
		assembler.Register<ChallengeSystem, ChallengeSystem>().Allocator(new HeapAllocator<ChallengeSystem>()).Binding(Binding.Scope);
		assembler.Register<ChallengeOverrides>().Allocator(new HeapAllocator<ChallengeOverrides>()).Binding(Binding.Scope);
		ChallengeDatabase instance7 = AssetBundleUtility.LoadAsset<ChallengeDatabase>("core", "ChallengeDatabase");
		assembler.Register<ChallengeDatabase, ChallengeDatabase>().Allocator(new SingletonAllocator<ChallengeDatabase>(instance7)).Binding(Binding.Scope);
		PlayTogetherChallengeDatabase instance8 = AssetBundleUtility.LoadAsset<PlayTogetherChallengeDatabase>("core", "PlayTogetherChallengeDatabase");
		assembler.Register<PlayTogetherChallengeDatabase, PlayTogetherChallengeDatabase>().Allocator(new SingletonAllocator<PlayTogetherChallengeDatabase>(instance8)).Binding(Binding.Scope);
		MapDatabase instance9 = AssetBundleUtility.LoadAsset<MapDatabase>("core", "MapDatabase");
		assembler.Register<MapDatabase, MapDatabase>().Allocator(new SingletonAllocator<MapDatabase>(instance9)).Binding(Binding.Scope);
		NewContentData instance10 = AssetBundleUtility.LoadAsset<NewContentData>("core", "NewContentData");
		assembler.Register<NewContentData>().Allocator(new SingletonAllocator<NewContentData>(instance10)).Binding(Binding.Scope);
		NewsAndNotificationData instance11 = AssetBundleUtility.LoadAsset<NewsAndNotificationData>("core", "NewsAndNotificationsData");
		assembler.Register<NewsAndNotificationData, NewsAndNotificationData>().Allocator(new SingletonAllocator<NewsAndNotificationData>(instance11)).Binding(Binding.Scope);
		CombinedMeshMaterials instance12 = AssetBundleUtility.LoadAsset<CombinedMeshMaterials>("core", "CombinedMeshMaterials");
		assembler.Register<CombinedMeshMaterials, CombinedMeshMaterials>().Allocator(new SingletonAllocator<CombinedMeshMaterials>(instance12)).Binding(Binding.Scope);
		assembler.Register<InputEvent, InputEvent>().Allocator(new ObjectPool<InputEvent>
		{
			InitialSize = 50,
			BlockSize = 50
		});
		assembler.Register<MotorwaysUIInputEvent, MotorwaysUIInputEvent>().Allocator(new ObjectPool<MotorwaysUIInputEvent>
		{
			InitialSize = 50,
			BlockSize = 50
		});
		assembler.Register<AxisInputEvent, AxisInputEvent>().Allocator(new ObjectPool<AxisInputEvent>
		{
			InitialSize = 50,
			BlockSize = 50
		});
		assembler.Register<FontDatabase>().Allocator(new GameObjectAllocator<FontDatabase>("core", "FontDatabase")).Binding(Binding.Scope);
		assembler.Register<IInitialGameScreen, LoadingScreen>().Allocator(new GameObjectPool<LoadingScreen>("core", "LoadingScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<StartupScreen>().Allocator(new GameObjectPool<StartupScreen>("core", "StartupScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<DeepLinkProcessor>().Allocator(new SingletonAllocator<DeepLinkProcessor>(new DeepLinkProcessor())).Binding(Binding.Scope);
		GameObject prefab = AssetBundleUtility.LoadPrefab("core", "MainMenuScreen");
		assembler.Register<MainMenuScreen>().Allocator(new GameObjectPool<MainMenuScreen>(prefab)
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<OptionsScreenMain>().Allocator(new GameObjectPool<OptionsScreenMain>("core", "OptionsScreenMain")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<OptionsScreenPause>().Allocator(new GameObjectPool<OptionsScreenPause>("core", "OptionsScreenPause")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<MapSelectScreen>().Allocator(new GameObjectPool<MapSelectScreen>("core", "MapSelectScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ResumeGameScreen>().Allocator(new GameObjectPool<ResumeGameScreen>("core", "ResumeGameScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<GameContainerScreen>().Allocator(new GameObjectPool<GameContainerScreen>("core", "GameContainerScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		GameObject prefab2 = AssetBundleUtility.LoadPrefab("core", $"GameOverScreen-{AppContainer.Environment.DeviceCategory}");
		assembler.Register<GameOverScreen>().Allocator(new GameObjectPool<GameOverScreen>(prefab2)
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<GameUpgradeScreen>().Allocator(new GameObjectPool<GameUpgradeScreen>("core", $"GameUpgradeScreen-{AppContainer.Environment.DeviceCategory}")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		GameObject prefab3 = AssetBundleUtility.LoadPrefab("core", "PauseScreen");
		assembler.Register<PauseScreen>().Allocator(new GameObjectPool<PauseScreen>(prefab3)
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<PhotoScreen>().Allocator(new GameObjectPool<PhotoScreen>("core", "PhotoScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<CinematicModeScreen>().Allocator(new GameObjectPool<CinematicModeScreen>("core", "CinematicModeScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ChallengeInfoScreen>().Allocator(new GameObjectPool<ChallengeInfoScreen>("core", "ChallengeInfoScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ProfileSelectScreen>().Allocator(new GameObjectPool<ProfileSelectScreen>("core", "ProfileSelectScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ProfileSelectButton>().Allocator(new GameObjectAllocator<ProfileSelectButton>("core", "ProfileSelectButton"));
		assembler.Register<ProfileCreationScreen>().Allocator(new GameObjectPool<ProfileCreationScreen>("core", "ProfileCreationScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<MovieScreen>().Allocator(new GameObjectPool<MovieScreen>("core", "MovieScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ExamplePopup>().Allocator(new GameObjectPool<ExamplePopup>("core", "ExamplePopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ChallengeInfoPopup>().Allocator(new GameObjectPool<ChallengeInfoPopup>("core", "ChallengeInfoPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ModeInfoPopup>().Allocator(new GameObjectPool<ModeInfoPopup>("core", "ModeInfoPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ModeInfoPopupInGame>().Allocator(new GameObjectPool<ModeInfoPopupInGame>("core", "ModeInfoPopupInGame")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ConfirmationPopup>().Allocator(new GameObjectPool<ConfirmationPopup>("core", "ConfirmationPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<CrossSavePopup>().Allocator(new GameObjectPool<CrossSavePopup>("core", "CrossSavePopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ColorblindCustomisePopup>().Allocator(new GameObjectPool<ColorblindCustomisePopup>("core", "ColorblindCustomisePopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<GenericPopup>().Allocator(new GameObjectPool<GenericPopup>("core", "GenericPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<LoadScreenInterruptionPopup>().Allocator(new GameObjectPool<LoadScreenInterruptionPopup>("core", "LoadScreenInterruptionPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<ExpertUnlockInfoPopup>().Allocator(new GameObjectPool<ExpertUnlockInfoPopup>("core", "ExpertUnlockInfoPopup")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		if (FeatureToggle.IsFeatureEnabled(Feature.AppleStoreDemo))
		{
			assembler.Register<AppleDemoCardPopup>().Allocator(new GameObjectPool<AppleDemoCardPopup>("demo", "AppleDemoCardPopup")
			{
				InitialSize = 1,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.ArcadeIntro))
		{
			assembler.Register<ArcadeIntroEndOfContentScreen>().Allocator(new GameObjectPool<ArcadeIntroEndOfContentScreen>("core", "ArcadeIntroEndOfContentScreen")
			{
				InitialSize = 1,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
		}
		assembler.Register<DebugOverlayScreen>().Allocator(new GameObjectPool<DebugOverlayScreen>("core", "DebugOverlayScreen")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<InGameMessage>().Allocator(new GameObjectPool<InGameMessage>("core", "InGameMessage")
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<MenuPlacementDefinition>().Allocator(new GameObjectAllocator<MenuPlacementDefinition>("core", "MenuDefinition")).Binding(Binding.Scope);
		assembler.Register<RoadTileAtlas>().Allocator(new HeapAllocator<RoadTileAtlas>()).Binding(Binding.Scope);
		assembler.Register<RoadTileSignature>().Allocator(new ObjectPool<RoadTileSignature>());
		assembler.Register<RoadTileDefinition>().Allocator(new ObjectPool<RoadTileDefinition>());
		assembler.Register<RoadTilePath>().Allocator(new ObjectPool<RoadTilePath>());
		assembler.Register<RoadTilePath.Piece>().Allocator(new ObjectPool<RoadTilePath.Piece>());
		assembler.Register<RoadTileMesh>().Allocator(new ObjectPool<RoadTileMesh>());
		assembler.Register<RoadTileConnectionStrokePath>().Allocator(new ObjectPool<RoadTileConnectionStrokePath>());
		assembler.Register<RailTileAtlas>().Allocator(new HeapAllocator<RailTileAtlas>()).Binding(Binding.Scope);
		assembler.Register<RailTileDefinition>().Allocator(new ObjectPool<RailTileDefinition>());
		assembler.Register<BoatPathTileAtlas>().Allocator(new HeapAllocator<BoatPathTileAtlas>()).Binding(Binding.Scope);
		assembler.Register<BoatPathTileDefinition>().Allocator(new ObjectPool<BoatPathTileDefinition>());
		assembler.Register<MenuNavigation, MotorwaysInGameStateToggleController>().Allocator(new HeapAllocator<MotorwaysInGameStateToggleController>()).Binding(Binding.Scope);
		assembler.Register<ILegacyUserProfile, LegacyMotorwaysUserProfile>().Allocator(new HeapAllocator<LegacyMotorwaysUserProfile>());
		assembler.Register<IExtendedUserProfile, MotorwaysExtendedUserProfile>().Allocator(new HeapAllocator<MotorwaysExtendedUserProfile>());
		assembler.Register<IDeviceSettings, MotorwaysDeviceSettings>().Allocator(new HeapAllocator<MotorwaysDeviceSettings>());
		assembler.Register<IGameJournalSave, MotorwaysGameJournalSave>().Allocator(new HeapAllocator<MotorwaysGameJournalSave>());
		assembler.Register<IMotorwaysGameJournalHeader, MotorwaysGameJournalHeader>().Allocator(new HeapAllocator<MotorwaysGameJournalHeader>());
		assembler.Register<InGameInputStateChangeAction>().Allocator(new ObjectPool<InGameInputStateChangeAction>
		{
			InitialSize = 50
		});
		assembler.Register<AchievementDefinition, MotorwaysAchievementDefinition>().Allocator(new HeapAllocator<MotorwaysAchievementDefinition>());
		assembler.Register<Achievement, MotorwaysAchievement>().Allocator(new HeapAllocator<MotorwaysAchievement>());
		assembler.Register<MotorwaysCityStatistics>().Allocator(new HeapAllocator<MotorwaysCityStatistics>());
		assembler.Register<MotorwaysTimedChallengeScore>().Allocator(new HeapAllocator<MotorwaysTimedChallengeScore>());
		assembler.Register<LeaderboardService>().Allocator(new HeapAllocator<LeaderboardService>()).Binding(Binding.Scope);
		if (FeatureToggle.IsFeatureEnabled(Feature.Analytics))
		{
			assembler.Register<AnalyticsEventHandler>().Allocator(new SingletonAllocator<AnalyticsEventHandler>(new AnalyticsEventHandler())).Binding(Binding.Scope);
		}
		NotificationDescriptorDatabase instance13 = AssetBundleUtility.LoadAsset<NotificationDescriptorDatabase>("core", "GameNotificationDatabase");
		assembler.Register<NotificationDescriptorDatabase>().Allocator(new SingletonAllocator<NotificationDescriptorDatabase>(instance13)).Binding(Binding.Scope);
		SupportedLocaleDatabase instance14 = AssetBundleUtility.LoadAsset<SupportedLocaleDatabase>("core", "SupportedLocaleDatabase");
		assembler.Register<SupportedLocaleDatabase, SupportedLocaleDatabase>().Allocator(new SingletonAllocator<SupportedLocaleDatabase>(instance14)).Binding(Binding.Scope);
		if (FeatureToggle.IsFeatureEnabled(Feature.CycleLanguages))
		{
			assembler.Register<SetLanguageAction>().Allocator(new ObjectPool<SetLanguageAction>
			{
				InitialSize = 2
			});
		}
		return assembler;
	}

	protected override Assembler CreateGameAssembler(Assembler appAssembler)
	{
		Assembler assembler = new Assembler("motorways");
		assembler.IsValidatingObjectScrubbing = Application.isEditor;
		assembler.Register<PseudorandomGenerator>().Allocator(new ObjectPool<PseudorandomGenerator>
		{
			InitialSize = 1
		});
		assembler.Register<RoadTileSignature>().Allocator(new ObjectPool<RoadTileSignature>());
		assembler.Register<ISimulation, Simulation>().Allocator(new ObjectPool<Simulation>()).Binding(Binding.Scope);
		assembler.Register<CommandJournal>().Allocator(new ObjectPool<CommandJournal>()).Binding(Binding.Scope);
		assembler.Register<Clock>().Allocator(new HeapAllocator<Clock>()).Binding(Binding.Scope);
		assembler.Register<IClient, MotorwaysClient>().Allocator(new ObjectPool<MotorwaysClient>
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<Passage>().Allocator(new ObjectPool<Passage>
		{
			InitialSize = 10
		});
		assembler.Register<City>().Allocator(new ObjectPool<City>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TileEditor>().Allocator(new HeapAllocator<TileEditor>()).Binding(Binding.Scope);
		assembler.Register<Pathfinder>().Allocator(new HeapAllocator<Pathfinder>()).Binding(Binding.Scope);
		assembler.Register<TilePathfinder>().Allocator(new HeapAllocator<TilePathfinder>()).Binding(Binding.Scope);
		assembler.Register<GameRules>().Allocator(new HeapAllocator<GameRules>()).Binding(Binding.Scope);
		assembler.Register<EndlessGameRules>().Allocator(new HeapAllocator<EndlessGameRules>()).Binding(Binding.Scope);
		assembler.Register<ExpertGameRules>().Allocator(new HeapAllocator<ExpertGameRules>()).Binding(Binding.Scope);
		assembler.Register<CreativeGameRules>().Allocator(new HeapAllocator<CreativeGameRules>()).Binding(Binding.Scope);
		assembler.Register<TutorialGameRules>().Allocator(new HeapAllocator<TutorialGameRules>()).Binding(Binding.Scope);
		assembler.Register<BackgroundGameRules>().Allocator(new HeapAllocator<BackgroundGameRules>()).Binding(Binding.Scope);
		assembler.Register<MovieGameRules>().Allocator(new HeapAllocator<MovieGameRules>()).Binding(Binding.Scope);
		assembler.Register<CinematicGameRules>().Allocator(new HeapAllocator<CinematicGameRules>()).Binding(Binding.Scope);
		assembler.Register<SelectUpgradeCommand>().Allocator(new ObjectPool<SelectUpgradeCommand>());
		assembler.Register<InitCityCommand>().Allocator(new ObjectPool<InitCityCommand>());
		assembler.Register<EditTileCommand>().Allocator(new ObjectPool<EditTileCommand>());
		assembler.Register<ReserveTileCommand>().Allocator(new ObjectPool<ReserveTileCommand>());
		assembler.Register<RemoveHouseCommand>().Allocator(new ObjectPool<RemoveHouseCommand>());
		assembler.Register<RemoveDestinationCommand>().Allocator(new ObjectPool<RemoveDestinationCommand>());
		assembler.Register<RemoveCarparkCommand>().Allocator(new ObjectPool<RemoveCarparkCommand>());
		assembler.Register<ClearTileReservationsCommand>().Allocator(new ObjectPool<ClearTileReservationsCommand>());
		assembler.Register<SetPausedCommand>().Allocator(new ObjectPool<SetPausedCommand>());
		assembler.Register<AdvanceTutorialCommand>().Allocator(new ObjectPool<AdvanceTutorialCommand>());
		assembler.Register<SnapshotCommand>().Allocator(new ObjectPool<SnapshotCommand>());
		assembler.Register<AddRoadEdit>().Allocator(new ObjectPool<AddRoadEdit>
		{
			InitialSize = 20
		});
		assembler.Register<AddRoundaboutEdit>().Allocator(new ObjectPool<AddRoundaboutEdit>
		{
			InitialSize = 20
		});
		assembler.Register<AddRoadLineEdit>().Allocator(new ObjectPool<AddRoadLineEdit>
		{
			InitialSize = 20
		});
		assembler.Register<AlignDrivewayEdit>().Allocator(new ObjectPool<AlignDrivewayEdit>
		{
			InitialSize = 5
		});
		assembler.Register<AddMotorwayEdit>().Allocator(new ObjectPool<AddMotorwayEdit>
		{
			InitialSize = 2
		});
		assembler.Register<MothballMotorwayEdit>().Allocator(new ObjectPool<MothballMotorwayEdit>
		{
			InitialSize = 5
		});
		assembler.Register<ClearTileEdit>().Allocator(new ObjectPool<ClearTileEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RemoveTrafficLightEdit>().Allocator(new ObjectPool<RemoveTrafficLightEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RemoveMotorwaysEdit>().Allocator(new ObjectPool<RemoveMotorwaysEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RemoveUnbuiltMotorwaysEdit>().Allocator(new ObjectPool<RemoveUnbuiltMotorwaysEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RemovePassagesEdit>().Allocator(new ObjectPool<RemovePassagesEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RemoveRoundaboutEdit>().Allocator(new ObjectPool<RemoveRoundaboutEdit>
		{
			InitialSize = 5
		});
		assembler.Register<RestoreMothballedPassageEdit>().Allocator(new ObjectPool<RestoreMothballedPassageEdit>
		{
			InitialSize = 5
		});
		assembler.Register<AddTrafficLightEdit>().Allocator(new ObjectPool<AddTrafficLightEdit>
		{
			InitialSize = 20
		});
		assembler.Register<AddUnbuiltMotorwayEdit>().Allocator(new ObjectPool<AddUnbuiltMotorwayEdit>
		{
			InitialSize = 20
		});
		assembler.Register<TileMatrixInt>().Allocator(new ObjectPool<TileMatrixInt>
		{
			InitialSize = 20
		});
		assembler.Register<TileMatrixBool>().Allocator(new ObjectPool<TileMatrixBool>
		{
			InitialSize = 20
		});
		assembler.Register<ClockProcess>().Allocator(new ObjectPool<ClockProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<EfficiencyCalculationProcess>().Allocator(new ObjectPool<EfficiencyCalculationProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<LaneUpdateProcess>().Allocator(new ObjectPool<LaneUpdateProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<VehicleMovementProcess>().Allocator(new ObjectPool<VehicleMovementProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<ParkVehiclesProcess>().Allocator(new ObjectPool<ParkVehiclesProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BuildMotorwaysProcess>().Allocator(new ObjectPool<BuildMotorwaysProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BuildRoundaboutsProcess>().Allocator(new ObjectPool<BuildRoundaboutsProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TrafficLightAlternatingProcess>().Allocator(new ObjectPool<TrafficLightAlternatingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<DispatchVehiclesProcess>().Allocator(new ObjectPool<DispatchVehiclesProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<GenerateDemandProcess>().Allocator(new ObjectPool<GenerateDemandProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<FailureStateProcess>().Allocator(new ObjectPool<FailureStateProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<AchievementCheckingProcess>().Allocator(new ObjectPool<AchievementCheckingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<UpgradeAwardingProcess>().Allocator(new ObjectPool<UpgradeAwardingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<UpgradeChoice>().Allocator(new ObjectPool<UpgradeChoice>
		{
			InitialSize = 10
		});
		assembler.Register<VehiclePathfindingProcess>().Allocator(new ObjectPool<VehiclePathfindingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<IntersectionEvaluatingProcess>().Allocator(new ObjectPool<IntersectionEvaluatingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<ReleaseMothballedLanesProcess>().Allocator(new ObjectPool<ReleaseMothballedLanesProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<ReleaseMotorwaysProcess>().Allocator(new ObjectPool<ReleaseMotorwaysProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TilePermanenceUpdatingProcess>().Allocator(new ObjectPool<TilePermanenceUpdatingProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BuildingSpawningProcess>().Allocator(new ObjectPool<BuildingSpawningProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<VehicleSpawningProcess>().Allocator(new ObjectPool<VehicleSpawningProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TrainSpawningProcess>().Allocator(new ObjectPool<TrainSpawningProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TrainMovementProcess>().Allocator(new ObjectPool<TrainMovementProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BoatMovementProcess>().Allocator(new ObjectPool<BoatMovementProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<OpenTrainCrossingsProcess>().Allocator(new ObjectPool<OpenTrainCrossingsProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BoatSpawningProcess>().Allocator(new ObjectPool<BoatSpawningProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<TutorialProgressionProcess>().Allocator(new ObjectPool<TutorialProgressionProcess>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<CityModel>().Allocator(new ModelPool<CityModel>
		{
			InitialSize = 3
		}).Binding(Binding.Scope);
		assembler.Register<CityPlanModel>().Allocator(new ModelPool<CityPlanModel>
		{
			InitialSize = 3
		}).Binding(Binding.Scope);
		assembler.Register<CityPlanModel.ScheduledBuilding>().Allocator(new ObjectPool<CityPlanModel.ScheduledBuilding>
		{
			InitialSize = 50
		});
		assembler.Register<DemandModel>().Allocator(new ModelPool<DemandModel>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<BuildingPlacer>().Allocator(new ObjectPool<BuildingPlacer>
		{
			InitialSize = 1
		}).Binding(Binding.Scope);
		assembler.Register<TilemapModel>().Allocator(new ModelPool<TilemapModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<TileModel>().Allocator(new ModelPool<TileModel>
		{
			InitialSize = 400,
			GrowthStrategy = GrowthStrategy.OnDemand,
			BlockSize = 50
		});
		assembler.Register<Tile>().Allocator(new ObjectPool<Tile>
		{
			InitialSize = 800,
			GrowthStrategy = GrowthStrategy.OnDemand,
			BlockSize = 50
		});
		assembler.Register<TileCornerModel>().Allocator(new ModelPool<TileCornerModel>
		{
			InitialSize = 200,
			GrowthStrategy = GrowthStrategy.OnDemand,
			BlockSize = 50
		});
		assembler.Register<MotorwayModel>().Allocator(new ModelPool<MotorwayModel>
		{
			InitialSize = 20
		});
		assembler.Register<ClockModel>().Allocator(new ModelPool<ClockModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ScoreModel>().Allocator(new ModelPool<ScoreModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<UpgradeDatabaseModel>().Allocator(new ModelPool<UpgradeDatabaseModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ActiveChallengesModel>().Allocator(new ModelPool<ActiveChallengesModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<GameBehaviourModel>().Allocator(new ModelPool<GameBehaviourModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<SnapshotModel>().Allocator(new ModelPool<SnapshotModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<VehicleDispatchRecord>().Allocator(new ObjectPool<VehicleDispatchRecord>
		{
			InitialSize = 10,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<IntersectionDecisionDatabaseModel>().Allocator(new ObjectPool<IntersectionDecisionDatabaseModel>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<IntersectionEntryDecision>().Allocator(new ObjectPool<IntersectionEntryDecision>
		{
			InitialSize = 100,
			GrowthStrategy = GrowthStrategy.Block,
			IsValidatingObjectScrubbing = false
		});
		assembler.Register<IntersectionEntryVehicleContext>().Allocator(new ObjectPool<IntersectionEntryVehicleContext>
		{
			InitialSize = 1000,
			GrowthStrategy = GrowthStrategy.Block,
			IsValidatingObjectScrubbing = false
		});
		assembler.Register<VehicleModel>().Allocator(new ModelPool<VehicleModel>
		{
			InitialSize = 200,
			BlockSize = 20
		});
		if (FeatureToggle.IsFeatureEnabled(Feature.WhatTheCarEasterEgg))
		{
			assembler.Register<EasterEggModel>().Allocator(new ModelPool<EasterEggModel>()).Binding(Binding.Scope);
		}
		assembler.Register<HouseModel>().Allocator(new ModelPool<HouseModel>
		{
			InitialSize = 20,
			BlockSize = 20
		});
		assembler.Register<DestinationModel>().Allocator(new ModelPool<DestinationModel>
		{
			InitialSize = 20,
			BlockSize = 20
		});
		assembler.Register<CarparkModel>().Allocator(new ModelPool<CarparkModel>
		{
			InitialSize = 20,
			BlockSize = 20
		});
		assembler.Register<CarparkModel.ParkingSpace>().Allocator(new ObjectPool<CarparkModel.ParkingSpace>
		{
			InitialSize = 60,
			BlockSize = 60
		});
		assembler.Register<LaneModel>().Allocator(new ModelPool<LaneModel>
		{
			InitialSize = 200,
			BlockSize = 100
		});
		assembler.Register<RoadChunkModel>().Allocator(new ModelPool<RoadChunkModel>
		{
			InitialSize = 100,
			BlockSize = 50
		});
		assembler.Register<RoadChunkModel.InboundVehicle>().Allocator(new ObjectPool<RoadChunkModel.InboundVehicle>
		{
			InitialSize = 100,
			BlockSize = 50
		});
		assembler.Register<TrafficLightModel>().Allocator(new ModelPool<TrafficLightModel>
		{
			InitialSize = 20,
			BlockSize = 20
		});
		assembler.Register<TrainCrossingModel>().Allocator(new ModelPool<TrainCrossingModel>
		{
			InitialSize = 20,
			BlockSize = 20
		});
		assembler.Register<RoundaboutModel>().Allocator(new ModelPool<RoundaboutModel>
		{
			InitialSize = 5,
			BlockSize = 5
		});
		assembler.Register<PassageModel>().Allocator(new ModelPool<PassageModel>
		{
			InitialSize = 5,
			BlockSize = 5
		});
		assembler.Register<AnchoredMessageModel>().Allocator(new ModelPool<AnchoredMessageModel>
		{
			InitialSize = 5,
			BlockSize = 5
		});
		assembler.Register<TreeModel>().Allocator(new ModelPool<TreeModel>
		{
			InitialSize = 10,
			BlockSize = 5
		});
		assembler.Register<TrainLineModel>().Allocator(new ModelPool<TrainLineModel>
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<RailTileModel>().Allocator(new ModelPool<RailTileModel>
		{
			InitialSize = 20,
			BlockSize = 5
		});
		assembler.Register<TrainModel>().Allocator(new ModelPool<TrainModel>
		{
			InitialSize = 3,
			BlockSize = 3
		});
		assembler.Register<BoatPathModel>().Allocator(new ModelPool<BoatPathModel>
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<BoatPathTileModel>().Allocator(new ModelPool<BoatPathTileModel>
		{
			InitialSize = 20,
			BlockSize = 5
		});
		assembler.Register<BoatModel>().Allocator(new ModelPool<BoatModel>
		{
			InitialSize = 3,
			BlockSize = 3
		});
		assembler.Register<CameraView>().Allocator(new ObjectPool<CameraView>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<ClockView>().Allocator(new GameObjectPool<ClockView>("core", "ClockView")).Binding(Binding.Scope);
		assembler.Register<ScoreView>().Allocator(new GameObjectPool<ScoreView>("core", "ScoreView")).Binding(Binding.Scope);
		if (FeatureToggle.IsFeatureEnabled(Feature.WrapperGameUI))
		{
			assembler.Register<UpgradeBarClient, UpgradeBarWrapper>().Allocator(new NestedGameObjectAllocator<UpgradeBarWrapper, GameUIScreen>()).Binding(Binding.Scope);
			assembler.Register<UpgradeBarClientHorizontal>();
		}
		else if (AppContainer.Environment.DeviceCategory == DeviceCategory.Desktop)
		{
			assembler.Register<UpgradeBarClient, UpgradeBarClientHorizontal>().Allocator(new NestedGameObjectAllocator<UpgradeBarClientHorizontal, GameUIScreen>()).Binding(Binding.Scope);
		}
		else
		{
			assembler.Register<UpgradeBarClient>().Allocator(new NestedGameObjectAllocator<UpgradeBarClient, GameUIScreen>()).Binding(Binding.Scope);
		}
		assembler.Register<EditMenuPanel>().Allocator(new NestedGameObjectAllocator<EditMenuPanel, GameUIScreen>()).Binding(Binding.Scope);
		assembler.Register<ColourWidget>().Allocator(new NestedGameObjectAllocator<ColourWidget, GameUIScreen>()).Binding(Binding.Scope);
		assembler.Register<NotificationView>().Allocator(new ObjectPool<NotificationView>()).Binding(Binding.Scope);
		assembler.Register<ChallengeView>().Allocator(new ObjectPool<ChallengeView>()).Binding(Binding.Scope);
		assembler.Register<CitySpawningView>().Allocator(new GameObjectPool<CitySpawningView>("core", "CitySpawningView")
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<TilemapView>().Allocator(new GameObjectPool<TilemapView>("core", "CityMap")
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<DeadEndRoadView>().Allocator(new GameObjectPool<DeadEndRoadView>("core", "DeadEndRoad")
		{
			InitialSize = 50,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 10
		});
		assembler.Register<AnimatedRoadTileConnectionView>().Allocator(new GameObjectPool<AnimatedRoadTileConnectionView>("core", "AnimatedRoadTileConnection")
		{
			InitialSize = 10,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 5
		});
		assembler.Register<TileView>().Allocator(new GameObjectPool<TileView>("core", "Tile")
		{
			InitialSize = 300,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 20
		});
		assembler.Register<TileSelectedView>().Allocator(new GameObjectPool<TileSelectedView>("core", "TileSelected")
		{
			InitialSize = 30,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<CombinedMeshThemeComponent>().Allocator(new SingletonAllocator<CombinedMeshThemeComponent>(new CombinedMeshThemeComponent()));
		VehicleMeshCombiner vehicleMeshCombiner = new VehicleMeshCombiner(AssetBundleUtility.LoadPrefab("core", "Vehicle"));
		appAssembler.Register<VehicleMeshCombiner>().Allocator(new SingletonAllocator<VehicleMeshCombiner>(vehicleMeshCombiner));
		assembler.Register<VehicleView>().Allocator(new GameObjectPool<VehicleView>(vehicleMeshCombiner.combinedMeshVehiclePrefab)
		{
			InitialSize = 200,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 20
		});
		if (FeatureToggle.IsFeatureEnabled(Feature.WhatTheCarEasterEgg))
		{
			assembler.Register<TribandVehicleEffects>().Allocator(new GameObjectPool<TribandVehicleEffects>("core", "TribandVehicleEffects"));
		}
		HouseMeshCombiner houseMeshCombiner = new HouseMeshCombiner(AssetBundleUtility.LoadPrefab("core", "House"));
		appAssembler.Register<HouseMeshCombiner>().Allocator(new SingletonAllocator<HouseMeshCombiner>(houseMeshCombiner)).Binding(Binding.Scope);
		assembler.Register<HouseView>().Allocator(new GameObjectPool<HouseView>(houseMeshCombiner.combinedMeshHousePrefab)
		{
			InitialSize = 100,
			GrowthStrategy = GrowthStrategy.Block,
			BlockSize = 10
		});
		assembler.Register<IndicatorAnimationView>().Allocator(new GameObjectPool<IndicatorAnimationView>("core", "IndicatorAnimations")
		{
			InitialSize = 10,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<PinView>().Allocator(new GameObjectPool<PinView>("core", "Pin")
		{
			InitialSize = 100,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<AnchoredMessageView>().Allocator(new GameObjectPool<AnchoredMessageView>("core", "AnchoredMessage")
		{
			InitialSize = 5,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		GameObject gameObject = AssetBundleUtility.LoadPrefab("core", "Destination");
		appAssembler.Register<DestinationMeshCombiner>().Allocator(new SingletonAllocator<DestinationMeshCombiner>(new DestinationMeshCombiner(gameObject))).Binding(Binding.Scope);
		assembler.Register<DestinationView>().Allocator(new GameObjectPool<DestinationView>(gameObject)
		{
			InitialSize = 40,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		CarparkMeshCombiner carparkMeshCombiner = new CarparkMeshCombiner(AssetBundleUtility.LoadPrefab("core", "Carpark"));
		appAssembler.Register<CarparkMeshCombiner>().Allocator(new SingletonAllocator<CarparkMeshCombiner>(carparkMeshCombiner)).Binding(Binding.Scope);
		assembler.Register<CarparkView>().Allocator(new GameObjectPool<CarparkView>(carparkMeshCombiner.combinedCarparkPrefab)
		{
			InitialSize = 30,
			GrowthStrategy = GrowthStrategy.OnDemand
		});
		assembler.Register<CombinedMeshView>().Allocator(new GameObjectPool<CombinedMeshView>("core", "CombinedMeshView")
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<RoadView>().Allocator(new GameObjectPool<RoadView>("core", "Road")
		{
			InitialSize = 500,
			BlockSize = 100
		});
		assembler.Register<MotorwayView>().Allocator(new GameObjectPool<MotorwayView>("core", "Motorway")
		{
			InitialSize = 20,
			BlockSize = 10
		});
		assembler.Register<TrafficLightView>().Allocator(new GameObjectPool<TrafficLightView>("core", "TrafficLight")
		{
			InitialSize = 10,
			BlockSize = 10
		});
		assembler.Register<UnbuiltMotorwayView>().Allocator(new GameObjectPool<UnbuiltMotorwayView>("core", "UnbuiltMotorway")
		{
			InitialSize = 10,
			BlockSize = 10
		});
		assembler.Register<RoundaboutView>().Allocator(new GameObjectPool<RoundaboutView>("core", "Roundabout")
		{
			InitialSize = 10,
			BlockSize = 10
		});
		assembler.Register<TreeView>().Allocator(new GameObjectPool<TreeView>("core", "Tree")
		{
			InitialSize = 20,
			BlockSize = 10
		});
		assembler.Register<RailView>().Allocator(new GameObjectPool<RailView>("core", "Rail")
		{
			InitialSize = 50,
			BlockSize = 10
		});
		assembler.Register<TrainCrossingView>().Allocator(new GameObjectPool<TrainCrossingView>("core", "TrainCrossing")
		{
			InitialSize = 5,
			BlockSize = 5
		});
		assembler.Register<TrainView>().Allocator(new GameObjectPool<TrainView>("core", "Train")
		{
			InitialSize = 3,
			BlockSize = 3
		});
		assembler.Register<BoatPathView>().Allocator(new GameObjectPool<BoatPathView>("core", "BoatPath")
		{
			InitialSize = 50,
			BlockSize = 10
		});
		assembler.Register<BoatView>().Allocator(new GameObjectPool<BoatView>("core", "Boat")
		{
			InitialSize = 1,
			BlockSize = 1
		});
		assembler.Register<ViewIndex>().Allocator(new ObjectPool<ViewIndex>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<AlertView>().Allocator(new GameObjectPool<AlertView>("core", "Alert")
		{
			InitialSize = 30
		});
		assembler.Register<BuildingIndicatorEventView>().Allocator(new ObjectPool<BuildingIndicatorEventView>
		{
			InitialSize = 20
		});
		assembler.Register<IndicatorEchoView>().Allocator(new GameObjectPool<IndicatorEchoView>("core", "IndicatorEcho")
		{
			InitialSize = 30
		});
		assembler.Register<IndicatorArrowView>().Allocator(new GameObjectPool<IndicatorArrowView>("core", "IndicatorArrow")
		{
			InitialSize = 30
		});
		assembler.Register<UpgradeCursor>().Allocator(new GameObjectPool<UpgradeCursor>("core", "UpgradeCursor")
		{
			InitialSize = 30
		});
		if (Application.isPlaying)
		{
			GameObject gameObject2 = new GameObject("City Schedule Debug View");
			gameObject2.SetActive(value: false);
			gameObject2.AddComponent<CityScheduleView>();
			assembler.Register<CityScheduleView>().Allocator(new GameObjectPool<CityScheduleView>(gameObject2)
			{
				InitialSize = 2,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
			gameObject2 = new GameObject("Simulation Toggle Debug View");
			gameObject2.AddComponent<SimulationToggleDebugView>();
			assembler.Register<SimulationToggleDebugView>().Allocator(new GameObjectPool<SimulationToggleDebugView>(gameObject2)
			{
				InitialSize = 2,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
			gameObject2 = new GameObject("Hotkey Debug View");
			gameObject2.AddComponent<HotkeyDebugView>();
			gameObject2.SetActive(value: false);
			appAssembler.Register<HotkeyDebugView>().Allocator(new GameObjectPool<HotkeyDebugView>(gameObject2)
			{
				InitialSize = 1,
				GrowthStrategy = GrowthStrategy.OnDemand
			}).Binding(Binding.Scope);
			gameObject2 = new GameObject("Tutorial Debug View");
			gameObject2.AddComponent<TutorialDebugView>();
			assembler.Register<TutorialDebugView>().Allocator(new GameObjectPool<TutorialDebugView>(gameObject2)
			{
				InitialSize = 2,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
			gameObject2 = new GameObject("Idle Vehicle Checker View");
			gameObject2.AddComponent<IdleVehicleCheckerDebugView>();
			assembler.Register<IdleVehicleCheckerDebugView>().Allocator(new GameObjectPool<IdleVehicleCheckerDebugView>(gameObject2)
			{
				InitialSize = 2,
				GrowthStrategy = GrowthStrategy.OnDemand
			});
		}
		assembler.Register<NetworkConnectivityUpdater>().Allocator(new ObjectPool<NetworkConnectivityUpdater>
		{
			InitialSize = 2
		}).Binding(Binding.Scope);
		assembler.Register<ClientUpgradeDatabase>().Allocator(new ObjectPool<ClientUpgradeDatabase>
		{
			InitialSize = 1,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		assembler.Register<BuildingsIndicatorView>().Allocator(new GameObjectPool<BuildingsIndicatorView>("core", "BuildingsIndicatorPrefab")
		{
			InitialSize = 2,
			GrowthStrategy = GrowthStrategy.OnDemand
		}).Binding(Binding.Scope);
		if (FeatureToggle.IsFeatureEnabled(Feature.WrapperGameUI))
		{
			assembler.Register<GameUIScreen, GameUIScreenWrapper>().Allocator(new GameObjectPool<GameUIScreenWrapper>("core", "InGameUI-Wrapper")
			{
				InitialSize = 2,
				GrowthStrategy = GrowthStrategy.OnDemand
			}).Binding(Binding.Scope);
		}
		else
		{
			assembler.Register<GameUIScreen>().Allocator(new GameObjectPool<GameUIScreen>("core", $"InGameUI-{AppContainer.Environment.DeviceCategory}")
			{
				InitialSize = 1,
				GrowthStrategy = GrowthStrategy.OnDemand
			}).Binding(Binding.Scope);
		}
		assembler.Register<NewRoadPreview>().Allocator(new GameObjectPool<NewRoadPreview>("core", "NewRoadPreview")
		{
			InitialSize = 2
		});
		assembler.Register<NewUpgradeAnimationView>().Allocator(new GameObjectPool<NewUpgradeAnimationView>("core", "NewUpgradeAnimation"));
		assembler.Register<AdvanceTutorialAction>().Allocator(new ObjectPool<AdvanceTutorialAction>
		{
			InitialSize = 2
		});
		assembler.Register<ToggleDrawModeAction>().Allocator(new ObjectPool<ToggleDrawModeAction>
		{
			InitialSize = 2
		});
		assembler.Register<DoubleTapToggleDrawModeAction>().Allocator(new ObjectPool<DoubleTapToggleDrawModeAction>
		{
			InitialSize = 2
		});
		assembler.Register<PressUIFocusAction>().Allocator(new ObjectPool<PressUIFocusAction>
		{
			InitialSize = 2
		});
		assembler.Register<ChangeGameSpeedAction>().Allocator(new ObjectPool<ChangeGameSpeedAction>
		{
			InitialSize = 2
		});
		assembler.Register<ChangeUpgradeBarAction>().Allocator(new ObjectPool<ChangeUpgradeBarAction>
		{
			InitialSize = 2
		});
		assembler.Register<LaneCursor>().Allocator(new ObjectPool<LaneCursor>
		{
			InitialSize = 5
		});
		assembler.Register<DrawRoadAction>().Allocator(new ObjectPool<DrawRoadAction>
		{
			InitialSize = 2
		});
		assembler.Register<ToggleDragClearTileAction>().Allocator(new ObjectPool<ToggleDragClearTileAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragClearTileAction>().Allocator(new ObjectPool<DragClearTileAction>
		{
			InitialSize = 2
		});
		assembler.Register<MoveInGameFocusAction>().Allocator(new ObjectPool<MoveInGameFocusAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragMoveInGameFocusAction>().Allocator(new ObjectPool<DragMoveInGameFocusAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDrawRoadAction>().Allocator(new ObjectPool<ControllerDrawRoadAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragMotorwayAction>().Allocator(new ObjectPool<DragMotorwayAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragMotorwayAction>().Allocator(new ObjectPool<ControllerDragMotorwayAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragEditMotorwayAction>().Allocator(new ObjectPool<ControllerDragEditMotorwayAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragCreativeModeEditableObjectAction>().Allocator(new ObjectPool<DragCreativeModeEditableObjectAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragTrafficLightAction>().Allocator(new ObjectPool<DragTrafficLightAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragTrafficLightAction>().Allocator(new ObjectPool<ControllerDragTrafficLightAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragRoundaboutAction>().Allocator(new ObjectPool<DragRoundaboutAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragRoundaboutAction>().Allocator(new ObjectPool<ControllerDragRoundaboutAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragMotorwayHandleAction>().Allocator(new ObjectPool<DragMotorwayHandleAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragMotorwayHandleAction>().Allocator(new ObjectPool<ControllerDragMotorwayHandleAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragEditMotorwayAction>().Allocator(new ObjectPool<DragEditMotorwayAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragHouseAction>().Allocator(new ObjectPool<DragHouseAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragHouseAction>().Allocator(new ObjectPool<ControllerDragHouseAction>
		{
			InitialSize = 2
		});
		assembler.Register<DragDestinationAction>().Allocator(new ObjectPool<DragDestinationAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerDragDestinationAction>().Allocator(new ObjectPool<ControllerDragDestinationAction>
		{
			InitialSize = 2
		});
		assembler.Register<ControllerEditMenuNavigateAction>().Allocator(new ObjectPool<ControllerEditMenuNavigateAction>
		{
			InitialSize = 2
		});
		assembler.Register<RemoteEditMenuNavigateAction>().Allocator(new ObjectPool<RemoteEditMenuNavigateAction>
		{
			InitialSize = 2
		});
		GameObject prefab = AssetBundleUtility.LoadPrefab("core", "DraftHouse");
		assembler.Register<DraftHouse>().Allocator(new GameObjectPool<DraftHouse>(prefab)
		{
			InitialSize = 1
		});
		GameObject prefab2 = AssetBundleUtility.LoadPrefab("core", "DraftDestination");
		assembler.Register<DraftDestination>().Allocator(new GameObjectPool<DraftDestination>(prefab2)
		{
			InitialSize = 1
		});
		assembler.Register<TouchCameraAction>().Allocator(new ObjectPool<TouchCameraAction>
		{
			InitialSize = 2
		});
		assembler.Register<ToggleZoomAction>().Allocator(new ObjectPool<ToggleZoomAction>
		{
			InitialSize = 2
		});
		assembler.Register<ToggleCreativeModeEditMenuAction>().Allocator(new ObjectPool<ToggleCreativeModeEditMenuAction>
		{
			InitialSize = 2
		});
		assembler.Register<OpenElectiveUpgradeScreenAction>().Allocator(new ObjectPool<OpenElectiveUpgradeScreenAction>
		{
			InitialSize = 2
		});
		assembler.Register<ToggleGameUIAction>().Allocator(new ObjectPool<ToggleGameUIAction>
		{
			InitialSize = 2
		});
		assembler.Register<MouseCameraAction>().Allocator(new ObjectPool<MouseCameraAction>
		{
			InitialSize = 2
		});
		assembler.Register<IGameStatistics, MotorwaysGameStatistics>().Allocator(new HeapAllocator<MotorwaysGameStatistics>());
		assembler.Register<ControllerCameraAction>().Allocator(new ObjectPool<ControllerCameraAction>
		{
			InitialSize = 2
		});
		if (FeatureToggle.IsFeatureEnabled(Feature.InGameDevTools))
		{
			assembler.Register<IInGameDevToolsRegistry, InGameDevToolsRegistry>().Allocator(new ObjectPool<InGameDevToolsRegistry>
			{
				InitialSize = 1
			}).Binding(Binding.Scope);
		}
		else
		{
			assembler.Register<IInGameDevToolsRegistry, NullInGameDevToolsRegistry>().Allocator(new ObjectPool<NullInGameDevToolsRegistry>
			{
				InitialSize = 1
			}).Binding(Binding.Scope);
		}
		assembler.Register<SimpleActionDevTool>().Allocator(new ObjectPool<SimpleActionDevTool>());
		assembler.Register<SimpleActionDevToolCommand>().Allocator(new ObjectPool<SimpleActionDevToolCommand>());
		assembler.Register<MotorwaysDevTool>().Allocator(new ObjectPool<MotorwaysDevTool>());
		assembler.Register<MotorwaysDevToolCommand>().Allocator(new ObjectPool<MotorwaysDevToolCommand>());
		assembler.Register<MotorwaysModelContainerTool>().Allocator(new ObjectPool<MotorwaysModelContainerTool>());
		assembler.Register<HouseDevTool>().Allocator(new ObjectPool<HouseDevTool>());
		assembler.Register<DestinationDevTool>().Allocator(new ObjectPool<DestinationDevTool>());
		assembler.Register<MotorwaysModelDevToolCommand>().Allocator(new ObjectPool<MotorwaysModelDevToolCommand>());
		AppContainer.Environment.PopulateGameAssembler(assembler);
		appAssembler.Register<Game, MotorwaysGame>().Allocator(new HeapAllocator<MotorwaysGame>()).EstablishScope(assembler)
			.Binding(Binding.EstablishedScope);
		assembler.Register<GameplayEventHandler>().Allocator(new ObjectPool<GameplayEventHandler>
		{
			InitialSize = 1
		}).Binding(Binding.Scope);
		return assembler;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
