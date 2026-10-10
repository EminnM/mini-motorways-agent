using Factory;
using Motorways.Models;
using Motorways.Processes;
using Server;

namespace Motorways.Commands;

[Serializable(1)]
public class InitCityCommand : Command
{
	[Dependency]
	private City _city;

	[Dependency]
	private IScope _scope;

	[Dependency]
	private BuildingPlacer _placer;

	[Dependency]
	private ChallengeDatabase _challengeDatabase;

	[Dependency]
	private MotorwaysGame _game;

	private string _cityName;

	[Serialize(false, null)]
	private CityDefinition _cityDefinition;

	private GameMode _mode;

	[Serialize(false, null)]
	private GameRules _rules;

	private ulong _seed;

	private MapChallenge.ChallengeType _challengeType;

	private int _cityChallengeIndex = -1;

	private ChallengeData[] _challenges;

	private int _challengeTimeStart;

	private int _challengeTimeEnd;

	public CityDefinition CityDefinition
	{
		get
		{
			return _cityDefinition;
		}
		set
		{
			if (Diagnostics.Verify(_cityDefinition == null, "Only set CityDefinition manually to fix up a deserialized InitCityCommand."))
			{
				_cityDefinition = value;
			}
		}
	}

	public string CityName => _cityName;

	public GameRules Rules
	{
		get
		{
			return _rules;
		}
		set
		{
			if (Diagnostics.Verify(_rules == null, "Only set GameRules manually to fix up a deserialized InitCityCommand."))
			{
				_rules = value;
			}
		}
	}

	public void Initialize(string cityName, CityDefinition cityDefinition, GameMode mode, GameRules rules, ulong seed)
	{
		_cityName = cityName;
		_cityDefinition = cityDefinition;
		_mode = mode;
		_rules = rules;
		_seed = seed;
	}

	private void SetChallengeData(MapChallenge.ChallengeType challengeType, int cityChallengeIndex, ChallengeData[] challenges, int timeStart, int timeEnd)
	{
		_challengeType = challengeType;
		_cityChallengeIndex = cityChallengeIndex;
		_challenges = challenges;
		_challengeTimeStart = timeStart;
		_challengeTimeEnd = timeEnd;
	}

	public override void Reset()
	{
		base.Reset();
		_cityName = "";
		_cityDefinition = null;
		_mode = GameMode.Normal;
		_rules = null;
		_seed = 0uL;
		_challengeType = MapChallenge.ChallengeType.None;
		_cityChallengeIndex = -1;
		_challenges = null;
		_challengeTimeStart = 0;
		_challengeTimeEnd = 0;
	}

	public override void Execute(ISimulation simulation)
	{
		CityModel cityModel = simulation.Scope.Get<CityModel>();
		cityModel.cityName = _cityName;
		cityModel.StartGameInMode(_mode, _rules);
		cityModel.pseudorandomGenerator = _scope.Get<PseudorandomGenerator>();
		cityModel.pseudorandomGenerator.Seed = _seed;
		simulation.AddModel(cityModel);
		if (_cityDefinition != null)
		{
			_city.Initialize(_cityDefinition, _rules);
		}
		simulation.AddProcess(_scope.Get<ClockProcess>());
		simulation.AddProcess(_scope.Get<BuildingSpawningProcess>());
		simulation.AddProcess(_scope.Get<BuildMotorwaysProcess>());
		simulation.AddProcess(_scope.Get<BuildRoundaboutsProcess>());
		simulation.AddProcess(_scope.Get<LaneUpdateProcess>());
		simulation.AddProcess(_scope.Get<GenerateDemandProcess>());
		simulation.AddProcess(_scope.Get<DispatchVehiclesProcess>());
		simulation.AddProcess(_scope.Get<TrafficLightAlternatingProcess>());
		simulation.AddProcess(_scope.Get<VehiclePathfindingProcess>());
		simulation.AddProcess(_scope.Get<IntersectionEvaluatingProcess>());
		simulation.AddProcess(_scope.Get<TrainSpawningProcess>());
		simulation.AddProcess(_scope.Get<BoatSpawningProcess>());
		simulation.AddProcess(_scope.Get<TrainMovementProcess>());
		simulation.AddProcess(_scope.Get<BoatMovementProcess>());
		simulation.AddProcess(_scope.Get<OpenTrainCrossingsProcess>());
		simulation.AddProcess(_scope.Get<VehicleSpawningProcess>());
		simulation.AddProcess(_scope.Get<VehicleMovementProcess>());
		simulation.AddProcess(_scope.Get<ParkVehiclesProcess>());
		simulation.AddProcess(_scope.Get<TutorialProgressionProcess>());
		simulation.AddProcess(_scope.Get<ReleaseMothballedLanesProcess>());
		simulation.AddProcess(_scope.Get<ReleaseMotorwaysProcess>());
		simulation.AddProcess(_scope.Get<TilePermanenceUpdatingProcess>());
		simulation.AddProcess(_scope.Get<EfficiencyCalculationProcess>());
		simulation.AddProcess(_scope.Get<FailureStateProcess>());
		simulation.AddProcess(_scope.Get<UpgradeAwardingProcess>());
		simulation.AddProcess(_scope.Get<AchievementCheckingProcess>());
		CityPlanModel model = simulation.Scope.Get<CityPlanModel>();
		simulation.AddModel(model);
		DemandModel model2 = simulation.Scope.Get<DemandModel>();
		simulation.AddModel(model2);
		TilemapModel model3 = simulation.Scope.Get<TilemapModel>();
		simulation.AddModel(model3);
		ClockModel model4 = simulation.Scope.Get<ClockModel>();
		simulation.AddModel(model4);
		ScoreModel model5 = simulation.Scope.Get<ScoreModel>();
		simulation.AddModel(model5);
		if (FeatureToggle.IsFeatureEnabled(Feature.ValidateSimulationDeterminism))
		{
			SnapshotModel model6 = simulation.Scope.Get<SnapshotModel>();
			simulation.AddModel(model6);
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.RecordIntersectionDecisions))
		{
			IntersectionDecisionDatabaseModel model7 = simulation.Scope.Get<IntersectionDecisionDatabaseModel>();
			simulation.AddModel(model7);
		}
		ActiveChallengesModel activeChallengesModel = simulation.Scope.Get<ActiveChallengesModel>();
		if (_challengeType != MapChallenge.ChallengeType.None && _rules.SupportsChallenges())
		{
			activeChallengesModel.challengeType = _challengeType;
			activeChallengesModel.cityChallengeIndex = _cityChallengeIndex;
			activeChallengesModel.timeEnd = _challengeTimeEnd;
			activeChallengesModel.timeStart = _challengeTimeStart;
			activeChallengesModel.challenges.AddRange(_challenges);
			activeChallengesModel.initialSeed = _seed;
		}
		else if (!(_cityDefinition is MockCityDefinition) && _challengeDatabase.debugInjectedChallenges != null && _challengeDatabase.debugInjectedChallenges.Count > 0 && _rules.SupportsChallenges() && FeatureToggle.IsFeatureEnabled(Feature.InjectDebugChallenges))
		{
			activeChallengesModel.challenges.AddRange(_challengeDatabase.debugInjectedChallenges);
		}
		UpgradeDatabaseModel upgradeDatabaseModel = simulation.Scope.Get<UpgradeDatabaseModel>();
		upgradeDatabaseModel.AwardStartingPackages();
		simulation.AddModel(upgradeDatabaseModel);
		simulation.AddModel(activeChallengesModel);
		GameBehaviourModel gameBehaviourModel = simulation.Scope.Get<GameBehaviourModel>();
		simulation.AddModel(gameBehaviourModel);
		if (gameBehaviourModel.OverridesGameModeWithExpert() && _mode != GameMode.Expert)
		{
			_game.ContinueInMode(GameMode.Expert);
		}
		if (_cityDefinition != null)
		{
			cityModel.startOffset = _cityDefinition.GenerateCityStartOffset(cityModel.pseudorandomGenerator);
			Command.Log.Info("Generated start offset {0}.", cityModel.startOffset);
			_placer.SetTileData(_city.Definition.TileWeightData);
			_city.GenerateCityLayout();
			_city.SetupTrainNetwork(simulation);
			_city.SetupBoatPathNetwork(simulation);
			_city.PopulateTrees(simulation);
			_cityDefinition.SetBonusTreeActiveState(gameBehaviourModel.UsesBonusTrees);
		}
		if (FeatureToggle.IsFeatureEnabled(Feature.WhatTheCarEasterEgg))
		{
			EasterEggModel model8 = simulation.Scope.Get<EasterEggModel>();
			simulation.AddModel(model8);
		}
	}

	public static InitCityCommand CreateNormalCity(IScope scope, string cityName, CityDefinition cityDefinition, GameMode mode, GameRules rules, uint seed)
	{
		InitCityCommand initCityCommand = scope.Get<InitCityCommand>();
		initCityCommand.Initialize(cityName, cityDefinition, mode, rules, seed);
		return initCityCommand;
	}

	public static InitCityCommand CreateChallengeCity(IScope scope, string cityName, CityDefinition cityDefinition, GameMode mode, GameRules rules, MapChallenge mapChallenge)
	{
		InitCityCommand initCityCommand = scope.Get<InitCityCommand>();
		initCityCommand.Initialize(cityName, cityDefinition, mode, rules, mapChallenge.seed);
		initCityCommand.SetChallengeData(mapChallenge.type, mapChallenge.cityChallengeIndex, mapChallenge.challenges, mapChallenge.TimeStart, mapChallenge.TimeEnd);
		return initCityCommand;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
