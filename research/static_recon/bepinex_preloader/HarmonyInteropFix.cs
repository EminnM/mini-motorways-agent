using System;
using System.Reflection;
using BepInEx.Bootstrap;
using BepInEx.Logging;
using HarmonyLib;
using HarmonyXInterop;

namespace BepInEx.Preloader.RuntimeFixes;

internal static class HarmonyInteropFix
{
	public static void Apply()
	{
		HarmonyInterop.Initialize(Paths.CachePath);
		Harmony.CreateAndPatchAll(typeof(HarmonyInteropFix), "org.bepinex.fixes.harmonyinterop");
	}

	[HarmonyReversePatch(HarmonyReversePatchType.Original)]
	[HarmonyPatch(typeof(Assembly), "LoadFile", new Type[] { typeof(string) })]
	private static Assembly LoadFile(string path)
	{
		return null;
	}

	[HarmonyPatch(typeof(Assembly), "LoadFile", new Type[] { typeof(string) })]
	[HarmonyPatch(typeof(Assembly), "LoadFrom", new Type[] { typeof(string) })]
	[HarmonyPrefix]
	private static bool OnAssemblyLoad(ref Assembly __result, string __0)
	{
		HarmonyInterop.TryShim(__0, Paths.GameRootPath, Logger.LogWarning, TypeLoader.ReaderParameters);
		__result = LoadFile(__0);
		return true;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
