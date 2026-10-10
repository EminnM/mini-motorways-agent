using System;
using System.IO;
using System.Reflection;
using BepInEx.Preloader;

namespace Doorstop;

internal static class Entrypoint
{
	private static string preloaderPath;

	/// <summary>
	///     The main entrypoint of BepInEx, called from Doorstop.
	/// </summary>
	public static void Start()
	{
		string text = $"preloader_{DateTime.Now:yyyyMMdd_HHmmss_fff}.log";
		try
		{
			EnvVars.LoadVars();
			text = Path.Combine(Path.GetDirectoryName(EnvVars.DOORSTOP_PROCESS_PATH) ?? ".", text);
			preloaderPath = Path.GetDirectoryName(Path.GetFullPath(EnvVars.DOORSTOP_INVOKE_DLL_PATH));
			AppDomain.CurrentDomain.AssemblyResolve += ResolveCurrentDirectory;
			typeof(Entrypoint).Assembly.GetType("BepInEx.Preloader.PreloaderRunner")?.GetMethod("PreloaderPreMain")?.Invoke(null, null);
		}
		catch (Exception ex)
		{
			File.WriteAllText(text, ex.ToString());
		}
		finally
		{
			AppDomain.CurrentDomain.AssemblyResolve -= ResolveCurrentDirectory;
		}
	}

	internal static Assembly ResolveCurrentDirectory(object sender, ResolveEventArgs args)
	{
		AssemblyName assemblyName = new AssemblyName(args.Name);
		try
		{
			return Assembly.LoadFile(Path.Combine(preloaderPath, assemblyName.Name + ".dll"));
		}
		catch (Exception)
		{
			return null;
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
