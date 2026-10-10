using System;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using BepInEx;
using UnityEngine;

namespace MiniMotorwaysInstrumentation
{
    [BepInPlugin(PluginGuid, PluginName, PluginVersion)]
    public class Plugin : BaseUnityPlugin
    {
        public const string PluginGuid = "local.research.minimotorways.instrumentation";
        public const string PluginName = "MiniMotorwaysInstrumentation";
        public const string PluginVersion = "0.1.0";

        // Dedicated log file, independent of BepInEx's own LogOutput.txt,
        // written into the research workspace (not the Steam install dir)
        // per the Phase 3 brief's requirement for logs/instrumentation.log.
        private static readonly string LogPath =
            @"C:\Users\TKA\Desktop\mini moto\logs\instrumentation.log";

        private static void WriteLine(string message)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(LogPath));
                string line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {message}";
                File.AppendAllText(LogPath, line + Environment.NewLine);
            }
            catch
            {
                // Never let logging failures affect the game itself.
            }
        }

        private void Awake()
        {
            try
            {
                int pid = Process.GetCurrentProcess().Id;
                string unityVersion = Application.unityVersion;
                string productName = Application.productName;
                string productVersion = Application.version;

                Assembly appAssembly = null;
                string appAssemblyInfo = "UNKNOWN";
                try
                {
                    appAssembly = Assembly.Load("App");
                    appAssemblyInfo = $"{appAssembly.FullName} (location={appAssembly.Location})";
                }
                catch (Exception ex)
                {
                    appAssemblyInfo = $"FAILED TO LOAD: {ex.Message}";
                }

                WriteLine("=== MiniMotorwaysInstrumentation plugin initializing ===");
                WriteLine($"Plugin version: {PluginVersion}");
                WriteLine($"Process ID: {pid}");
                WriteLine($"Unity version (Application.unityVersion): {unityVersion}");
                WriteLine($"Product: {productName} {productVersion}");
                WriteLine($"App.dll assembly info: {appAssemblyInfo}");
                WriteLine($"BepInEx Paths.GameRootPath: {Paths.GameRootPath}");
                WriteLine($"BepInEx Paths.PluginPath: {Paths.PluginPath}");
                WriteLine("=== Initialization complete ===");

                Logger.LogInfo($"{PluginName} v{PluginVersion} initialized. PID={pid}, Unity={unityVersion}. See instrumentation_logs/instrumentation.log for details.");
            }
            catch (Exception ex)
            {
                WriteLine($"EXCEPTION during Awake: {ex}");
                Logger.LogError($"Exception during {PluginName} Awake: {ex}");
            }
        }
    }
}
