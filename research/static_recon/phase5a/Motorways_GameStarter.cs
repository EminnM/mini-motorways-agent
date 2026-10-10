using System;
using System.IO;
using Factory;
using Motorways.Models;
using Motorways.Views;
using Screens;
using Server;
using UnityEngine;

namespace Motorways;

public class GameStarter
{
	private MonoBehaviour _coroutineHost;

	private AssetBundleUtility.AsyncLoadResult _cityDefinitionLoader;

	private MapDefinition _mapDefinition;

	private MapChallenge _mapChallenge;

	private GameMode _mode;

	private MotorwaysGameJournalSave _save;

	private bool _replaceTopScreen;

	private bool _skipTransitionIn;

	private bool _startPaused;

	private float _transitionInDuration;

	private ScreenStack.MotorwaysScreen _startScreen = ScreenStack.MotorwaysScreen.InGame;

	public bool CanStart => _cityDefinitionLoader?.HasValue ?? false;

	private bool UseCustomStartScreen
	{
		get
		{
			if (_startScreen != ScreenStack.MotorwaysScreen.Movie)
			{
				return _startScreen == ScreenStack.MotorwaysScreen.CinematicMode;
			}
			return true;
		}
	}

	public GameStarter(MonoBehaviour coroutineHost)
	{
		_coroutineHost = coroutineHost;
	}

	public bool StartFromSavedGame(MapLibrary mapLibrary, MotorwaysGameJournalSave save, bool replaceTopScreen = false, bool skipNextTransition = false, bool startPaused = false)
	{
		if (!LoadMapDefinition(mapLibrary.GetMapByName(save.CityId)))
		{
			return false;
		}
		_save = save;
		_replaceTopScreen = replaceTopScreen;
		_skipTransitionIn = skipNextTransition;
		_startPaused = startPaused;
		_startScreen = ScreenStack.MotorwaysScreen.InGame;
		return true;
	}

	public bool StartSavedGameFromCustomScreen(MapLibrary mapLibrary, MotorwaysGameJournalSave save, ScreenStack.MotorwaysScreen customScreen, bool skipNextTransition = false, bool startPaused = false)
	{
		if (!LoadMapDefinition(mapLibrary.GetMapByName(save.CityId)))
		{
			return false;
		}
		_save = save;
		_skipTransitionIn = skipNextTransition;
		_startScreen = customScreen;
		_startPaused = startPaused;
		return true;
	}

	public bool StartFromMapDefinition(MapDefinition mapDefinition, GameMode mode, float transitionInDuration = 0f, bool replaceTopScreen = false, bool startPaused = false)
	{
		if (!LoadMapDefinition(mapDefinition))
		{
			return false;
		}
		_replaceTopScreen = replaceTopScreen;
		_mode = mode;
		_save = null;
		_transitionInDuration = transitionInDuration;
		_startScreen = ScreenStack.MotorwaysScreen.InGame;
		_startPaused = startPaused;
		return true;
	}

	public bool Start(ScreenStack screenStack, IScope appScope)
	{
		if (!Diagnostics.Verify(CanStart, "GameStarter not in a valid state to start. You should check against GameStarter.CanStart before calling this."))
		{
			return false;
		}
		if (UseCustomStartScreen && _save == null)
		{
			Diagnostics.FailAssert("We can't load into a custom start screen with a fresh game!");
			return false;
		}
		CityDefinition cityDefinition = UnityEngine.Object.Instantiate(_cityDefinitionLoader.asset as GameObject).GetComponent<CityDefinition>();
		if (!screenStack.IsScreenActive(ScreenStack.MotorwaysScreen.MapSelect))
		{
			MapSelectScreen mapSelectScreen = appScope.Get<MapSelectScreen>();
			mapSelectScreen.SavePreviouslyLockedMaps();
			appScope.Release(mapSelectScreen);
		}
		if (_save == null)
		{
			if (_replaceTopScreen)
			{
				screenStack.ReplaceScreenOnTop(ScreenStack.MotorwaysScreen.InGame, delegate(GameContainerScreen newScreen)
				{
					if (_skipTransitionIn)
					{
						newScreen.SkipNextTransition();
					}
					else if (_transitionInDuration > 0f)
					{
						newScreen.OverrideNextTransition(_transitionInDuration);
					}
					newScreen.PrepareForMap(cityDefinition, _mapDefinition, _mode, _mapChallenge);
				});
			}
			else
			{
				IScreen topVisibleScreen = screenStack.GetTopVisibleScreen();
				if (!screenStack.IsScreenActive(ScreenStack.MotorwaysScreen.MainMenu))
				{
					screenStack.PushScreen<MainMenuScreen>(ScreenStack.MotorwaysScreen.MainMenu).gameObject.SetActive(value: false);
				}
				screenStack.PushScreen(ScreenStack.MotorwaysScreen.InGame, delegate(GameContainerScreen newScreen)
				{
					if (_skipTransitionIn)
					{
						newScreen.SkipNextTransition();
					}
					else if (_transitionInDuration > 0f)
					{
						newScreen.OverrideNextTransition(_transitionInDuration);
					}
					newScreen.PrepareForMap(cityDefinition, _mapDefinition, _mode, _mapChallenge);
				}, additive: false, null, blocksGameInput: true, topVisibleScreen);
			}
			return true;
		}
		Game game = _save.DeserializeGame(cityDefinition);
		MotorwaysGame motorwaysGame = game as MotorwaysGame;
		if (motorwaysGame != null)
		{
			motorwaysGame.FixDeserializedSimulation(cityDefinition);
			ISimulation simulation = motorwaysGame.Simulation;
			ModelListEnumerator<TileModel> enumerator = simulation.GetModels<TileModel>().GetEnumerator();
			while (enumerator.MoveNext())
			{
				TileModel current = enumerator.Current;
				if (current.Tile.ContentType != TileContentType.None)
				{
					Vector2Int coordinates = current.Coordinates;
					if (cityDefinition.TileIsOverWater(coordinates) || cityDefinition.TileIsUnderAMountain(coordinates))
					{
						Diagnostics.FailAssert("Deserialised simulation has a tile with {0} in an invalid location.", current.Tile.ContentType);
						motorwaysGame = null;
						break;
					}
				}
			}
			if (motorwaysGame != null)
			{
				int num = -1;
				ModelListEnumerator<HouseModel> enumerator2 = simulation.GetModels<HouseModel>().GetEnumerator();
				while (enumerator2.MoveNext())
				{
					HouseModel current2 = enumerator2.Current;
					num = Mathf.Max(num, current2.GroupIndex);
				}
				ModelListEnumerator<DestinationModel> enumerator3 = simulation.GetModels<DestinationModel>().GetEnumerator();
				while (enumerator3.MoveNext())
				{
					DestinationModel current3 = enumerator3.Current;
					num = Mathf.Max(num, current3.GroupIndex);
				}
				int count = cityDefinition.schedulePlanner.scheduleGroups.Count;
				if (num >= count)
				{
					Diagnostics.FailAssert("Deserialised simulation has a building with a group index of {0}, but the city only has {1}.", num, count);
					motorwaysGame = null;
				}
			}
			if (motorwaysGame != null)
			{
				bool isPaused = simulation.IsPaused;
				game.Simulation.IsPaused = true;
				try
				{
					simulation.Step();
					simulation.IsPaused = isPaused;
					game.Scope.Get<Clock>().Rewind();
				}
				catch (Exception ex)
				{
					Diagnostics.FailAssert("Deserialised simulation failed a paused step.\n{0}", ex);
					motorwaysGame = null;
				}
				if (motorwaysGame != null && _save.ChallengeType != MapChallenge.ChallengeType.None)
				{
					ActiveChallengesModel model = motorwaysGame.Simulation.GetModel<ActiveChallengesModel>();
					switch (_save.ChallengeType)
					{
					case MapChallenge.ChallengeType.City:
						_mapChallenge = MapChallenge.CreateCityChallenge(game.Scope.Get<ChallengeSystem>(), model.cityChallengeIndex, _mapDefinition, model.challenges.ToArray(), model.initialSeed);
						break;
					case MapChallenge.ChallengeType.Daily:
						_mapChallenge = MapChallenge.CreateDailyChallenge(game.Scope.Get<ChallengeSystem>(), _mapDefinition, model.challenges.ToArray(), model.timeStart, model.timeEnd, model.initialSeed);
						break;
					case MapChallenge.ChallengeType.Weekly:
						_mapChallenge = MapChallenge.CreateWeeklyChallenge(game.Scope.Get<ChallengeSystem>(), _mapDefinition, model.challenges.ToArray(), model.timeStart, model.timeEnd, model.initialSeed);
						break;
					case MapChallenge.ChallengeType.Mystery:
						_mapChallenge = MapChallenge.RebuildMysteryChallenge(game.Scope.Get<ChallengeSystem>(), _mapDefinition, model.challenges.ToArray(), model.initialSeed);
						break;
					default:
						Diagnostics.FailAssert($"Invalid ChallengeType for game save: {_save.ChallengeType}");
						break;
					}
				}
			}
		}
		if (motorwaysGame == null)
		{
			if (FeatureToggle.IsFeatureEnabled(Feature.DiagnosticReports))
			{
				Diagnostics.Report report = new Diagnostics.Report();
				report.Motive = "deserializeException";
				report.SetMetadata("buildName", Version.Name, index: true);
				report.SetMetadata("buildTimestamp", Version.Timestamp.ToString(), index: true);
				if (!string.IsNullOrEmpty(Version.CommitHash))
				{
					report.SetMetadata("commitHash", Version.CommitHash, index: true);
				}
				report.SetMetadata("city", _save.CityId, index: true);
				report.SetMetadata("gameMode", _save.Mode.ToString(), index: true);
				MemoryStream memoryStream = new MemoryStream();
				using (BinaryWriter binaryWriter = new BinaryWriter(memoryStream))
				{
					_save.OnSerializeBeforeData(binaryWriter);
					binaryWriter.Write(_save.GetBytesForSerializing());
				}
				report.AttachFile("simulation.gamejournal", memoryStream.ToArray());
				report.Upload();
			}
			UnityEngine.Object.Destroy(cityDefinition.gameObject);
			game?.Scope.Release(game);
			return false;
		}
		if (_replaceTopScreen)
		{
			screenStack.ReplaceScreenOnTop(_startScreen, delegate(BaseScalingScreen newScreen)
			{
				if (_skipTransitionIn)
				{
					newScreen.SkipNextTransition();
				}
				else if (_transitionInDuration > 0f)
				{
					newScreen.OverrideNextTransition(_transitionInDuration);
				}
				if (newScreen is IGameStartScreen gameStartScreen)
				{
					gameStartScreen.PrepareForNewGame(cityDefinition, _mapDefinition, motorwaysGame, _mapChallenge, _startPaused);
				}
				else
				{
					Diagnostics.FailAssert($"GameStarter attempting to start with unsupported ScreenStack.MotorwaysScreen: {_startScreen}");
				}
			}, motorwaysGame.Scope);
		}
		else
		{
			screenStack.PushScreen(_startScreen, delegate(BaseScalingScreen newScreen)
			{
				if (_skipTransitionIn)
				{
					newScreen.SkipNextTransition();
				}
				else if (_transitionInDuration > 0f)
				{
					newScreen.OverrideNextTransition(_transitionInDuration);
				}
				if (newScreen is IGameStartScreen gameStartScreen)
				{
					gameStartScreen.PrepareForNewGame(cityDefinition, _mapDefinition, motorwaysGame, _mapChallenge, _startPaused);
				}
				else
				{
					Diagnostics.FailAssert($"GameStarter attempting to start with unsupported ScreenStack.MotorwaysScreen: {_startScreen}");
				}
			}, additive: false, motorwaysGame.Scope);
		}
		return true;
	}

	private bool LoadMapDefinition(MapDefinition mapDefinition)
	{
		if (!Diagnostics.Verify(_cityDefinitionLoader == null, "City definition loader isn't null!"))
		{
			return false;
		}
		if (Diagnostics.Verify(mapDefinition != null))
		{
			_mapDefinition = mapDefinition;
		}
		_cityDefinitionLoader = AssetBundleUtility.LoadPrefabAsync(_mapDefinition.mapAssetBundle, _mapDefinition.mapPrefabName, _coroutineHost);
		return true;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
