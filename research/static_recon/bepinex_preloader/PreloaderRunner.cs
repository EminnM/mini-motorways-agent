using System;
using System.IO;
using System.Linq;
using System.Reflection;
using BepInEx.Preloader.RuntimeFixes;
using Doorstop;

namespace BepInEx.Preloader;

internal static class PreloaderRunner
{
	private static readonly string[] CriticalAssemblies = new string[7] { "Mono.Cecil.dll", "Mono.Cecil.Mdb.dll", "Mono.Cecil.Pdb.dll", "Mono.Cecil.Rocks.dll", "MonoMod.Utils.dll", "MonoMod.RuntimeDetour.dll", "0Harmony.dll" };

	private static void LoadCriticalAssemblies()
	{
		string[] criticalAssemblies = CriticalAssemblies;
		foreach (string path in criticalAssemblies)
		{
			try
			{
				Assembly.LoadFile(Path.Combine(Paths.BepInExAssemblyDirectory, path));
			}
			catch (Exception)
			{
			}
		}
	}

	public static void PreloaderPreMain()
	{
		PlatformUtils.SetPlatform();
		string bepinRootPath = Utility.ParentDirectory(Path.GetFullPath(EnvVars.DOORSTOP_INVOKE_DLL_PATH), 2);
		Paths.SetExecutablePath(EnvVars.DOORSTOP_PROCESS_PATH, bepinRootPath, EnvVars.DOORSTOP_MANAGED_FOLDER_DIR, EnvVars.DOORSTOP_DLL_SEARCH_DIRS);
		LoadCriticalAssemblies();
		AppDomain.CurrentDomain.AssemblyResolve += LocalResolve;
		AppDomain.CurrentDomain.AssemblyResolve -= Entrypoint.ResolveCurrentDirectory;
		PreloaderMain();
	}

	private static void PreloaderMain()
	{
		if (Preloader.ConfigApplyRuntimePatches.Value)
		{
			XTermFix.Apply();
			ConsoleSetOutFix.Apply();
		}
		Preloader.Run();
	}

	private static Assembly LocalResolve(object sender, ResolveEventArgs args)
	{
		if (!Utility.TryParseAssemblyName(args.Name, out var assemblyName))
		{
			return null;
		}
		AssemblyName assemblyName2;
		var source = Enumerable.OrderByDescending(from a in AppDomain.CurrentDomain.GetAssemblies()
			select new
			{
				assembly = a,
				name = (Utility.TryParseAssemblyName(a.FullName, out assemblyName2) ? assemblyName2 : null)
			} into a
			where a.name != null && a.name.Name == assemblyName.Name
			select a, a => a.name.Version).ToList();
		Assembly assembly = (source.FirstOrDefault(a => a.name.Version == assemblyName.Version) ?? source.FirstOrDefault())?.assembly;
		if ((object)assembly != null)
		{
			return assembly;
		}
		if (Utility.TryResolveDllAssembly(assemblyName, Paths.BepInExAssemblyDirectory, out assembly) || Utility.TryResolveDllAssembly(assemblyName, Paths.PatcherPluginPath, out assembly) || Utility.TryResolveDllAssembly(assemblyName, Paths.PluginPath, out assembly))
		{
			return assembly;
		}
		return null;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
