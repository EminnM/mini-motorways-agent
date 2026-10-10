using UnityEngine;
using UnityEngine.SceneManagement;

namespace Motorways;

public class CityBootstrap : MonoBehaviour
{
	public CityDefinition cityDefinition;

	private bool _hasStartedGame;

	[HideInInspector]
	public string _playbackAppJournalPath;

	private static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("CityBootstrap");

	private void Awake()
	{
		SceneManager.LoadScene(1, LoadSceneMode.Additive);
	}

	private void Update()
	{
		if (!_hasStartedGame && SceneManager.GetSceneByName("Runtime").IsValid())
		{
			EnableRuntime();
			_hasStartedGame = true;
		}
	}

	private void EnableRuntime()
	{
		AppRuntime appRuntime = Resources.FindObjectsOfTypeAll<AppRuntime>()[0];
		if (!string.IsNullOrEmpty(_playbackAppJournalPath))
		{
			appRuntime._playbackAppJournalPath = _playbackAppJournalPath;
		}
		appRuntime.gameObject.SetActive(value: true);
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
