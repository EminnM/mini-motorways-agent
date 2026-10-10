// PROPOSED plugin addition (NOT applied: editing game_plugin/DoorstopBootstrap/Entrypoint.cs needs the user's explicit permission).
// Insert the two methods below next to HandleIpcDiagSpawn in Entrypoint.cs and add this line to the IPC switch (Entrypoint.cs ~line 4089):
//     case "DIAG_FORK": return HandleIpcDiagFork(obj);
// Then: cd game_plugin/DoorstopBootstrap && dotnet build -c Release ; back up the deployed DLL; copy bin/Release/net472/MiniMotorwaysBootstrap.dll to <GameDir>/Bootstrap/.
// Everything reused (RestoreGameFromBytes, ExportSim, CaptureExtras, SpawnKeys, ApplyAction, AdvanceSimulation, IsTerminal, ActionIndexToDescriptor, JStr/JBool/Inv) already exists in the file.
//
// DIAG_FORK request:  {"op":"DIAG_FORK","n_pre":0,"horizon":120,"checkpoints":[30,60,90,120],"intervention_seconds":0.3,
//                      "branches":[{"name":"base","actions":[]},{"name":"base2","actions":[]},{"name":"block","actions":[1234,1238,1242]},{"name":"sham","actions":[...]}]}
// Response: {"op":"DIAG_FORK","nPre","horizon","bytes","prngAtSnapshot","concreteAtSnapshot","fingerprintTiles",
//            "branches":{"<name>":{"restoreNote","prngAtRestore","prngAfterEdits","concreteAtRestore","concreteAfterEdits","editsAccepted","edits":[{action,ok,info}],
//                                  "terminalStep","spawns":[[t,"H"|"D",group,x,y],...],"checkpoints":[{"t","houses","dests","prng","concrete","score"},...]}}}
// After the op the IPC game is the last restored branch: the Python caller RESETs before normal play (harness.py does).

            private string RunForkBranch(byte[] bytes, SnapshotExtras extras, string name, System.Collections.Generic.List<ActionDescriptor> edits, int horizon, int[] checkpoints, double interventionSeconds)
            {
                TearDownIpcGameForDiag();
                string note;
                MotorwaysGame g = RestoreGameFromBytes(bytes, out note, extras);
                _ipcGame = g;
                Func<string> prngOf = () => { try { return g.Simulation.GetModel<CityModel>().pseudorandomGenerator.ToString(); } catch (Exception) { return "?"; } };
                Func<int> concreteOf = () => { try { return g.Simulation.GetModel<UpgradeDatabaseModel>().GetAvailableUpgradeCount(UpgradeType.Concrete); } catch (Exception) { return -1; } };
                string prngRestore = prngOf();
                int concrete0 = concreteOf();
                string fpRestore = StateFingerprint(g).Split(';')[0];        // tiles hash right after restore, before any edit: must equal fingerprintTiles of the snapshot
                var seen = SpawnKeys(g);
                var spawns = new StringBuilder();
                int nSpawns = 0;
                Action<double> logNew = (tNow) =>
                {
                    var now = SpawnKeys(g);
                    var fresh = new System.Collections.Generic.List<string>();
                    foreach (string k in now) if (!seen.Contains(k)) fresh.Add(k);
                    fresh.Sort(StringComparer.Ordinal);
                    foreach (string k in fresh)
                    {
                        string[] p = k.Split(',');
                        if (nSpawns++ > 0) spawns.Append(",");
                        spawns.Append("[" + Inv(tNow) + ",\"" + p[0] + "\"," + p[1] + "," + p[2] + "," + p[3] + "]");
                        seen.Add(k);
                    }
                };
                var editLog = new StringBuilder();
                int acc = 0;
                if (edits.Count == 0) AdvanceSimulation(g, interventionSeconds);
                else
                {
                    foreach (var e in edits)
                    {
                        string info = "";
                        bool ok = false;
                        try { ok = ApplyAction(g, e, out info); } catch (Exception ex) { info = "EXC " + ex.Message; }
                        if (ok) acc++;
                        if (editLog.Length > 0) editLog.Append(",");
                        editLog.Append("{\"action\":" + JStr(e.ToString()) + ",\"ok\":" + JBool(ok) + ",\"info\":" + JStr(info ?? "") + "}");
                        AdvanceSimulation(g, interventionSeconds / edits.Count);
                    }
                }
                string prngAfterEdits = prngOf();
                int concreteAfterEdits = concreteOf();
                double clock0 = (double)g.Scope.Get<Clock>().Time;
                logNew(0.0);
                var cps = new StringBuilder();
                int termStep = -1;
                for (int step = 1; step <= horizon; step++)
                {
                    AdvanceSimulation(g, _ipcDecisionDeltaSeconds);
                    logNew((double)g.Scope.Get<Clock>().Time - clock0);
                    if (termStep < 0 && IsTerminal(g)) termStep = step;
                    for (int ci = 0; ci < checkpoints.Length; ci++)
                    {
                        if (checkpoints[ci] != step) continue;
                        int h = 0, d = 0;
                        ModelListEnumerator<HouseModel> eh = g.Simulation.GetModels<HouseModel>().GetEnumerator(); while (eh.MoveNext()) h++;
                        ModelListEnumerator<DestinationModel> ed = g.Simulation.GetModels<DestinationModel>().GetEnumerator(); while (ed.MoveNext()) d++;
                        if (cps.Length > 0) cps.Append(",");
                        cps.Append("{\"t\":" + step + ",\"houses\":" + h + ",\"dests\":" + d + ",\"prng\":" + JStr(prngOf()) + ",\"concrete\":" + concreteOf() + ",\"score\":" + Inv((double)g.Simulation.GetModel<ScoreModel>().Score) + "}");
                    }
                }
                return "\"" + name + "\":{\"restoreNote\":" + JStr(note ?? "") + ",\"fingerprintAtRestore\":" + JStr(fpRestore) + ",\"prngAtRestore\":" + JStr(prngRestore) + ",\"prngAfterEdits\":" + JStr(prngAfterEdits) + ",\"concreteAtRestore\":" + concrete0 + ",\"concreteAfterEdits\":" + concreteAfterEdits
                     + ",\"editsAccepted\":" + acc + ",\"edits\":[" + editLog + "],\"terminalStep\":" + termStep + ",\"spawns\":[" + spawns + "],\"checkpoints\":[" + cps + "]}";
            }

            private string HandleIpcDiagFork(System.Collections.Generic.Dictionary<string, object> obj)
            {
                try
                {
                    if (_ipcGame == null) return "{\"op\":\"DIAG_FORK\",\"error\":\"no_active_episode\"}";
                    Func<string, int, int> gi = (key, def) => obj != null && obj.ContainsKey(key) ? (int)(double)obj[key] : def;
                    int nPre = gi("n_pre", 0), horizon = gi("horizon", 120);
                    double interventionSeconds = obj != null && obj.ContainsKey("intervention_seconds") ? (double)obj["intervention_seconds"] : 0.3;
                    var cpl = new System.Collections.Generic.List<int>();
                    if (obj != null && obj.ContainsKey("checkpoints")) foreach (object o in (System.Collections.Generic.List<object>)obj["checkpoints"]) cpl.Add((int)(double)o);
                    if (cpl.Count == 0) cpl.Add(horizon);
                    int[] cps = cpl.ToArray();
                    _diagRestoreMode = 1; _diagUseExtras = true;
                    for (int i = 0; i < nPre; i++) ScriptedDiagStep(_ipcGame, i, -1);
                    string fp0 = StateFingerprint(_ipcGame);
                    string prng0 = "?";
                    try { prng0 = _ipcGame.Simulation.GetModel<CityModel>().pseudorandomGenerator.ToString(); } catch (Exception) { }
                    int concrete0 = -1;
                    try { concrete0 = _ipcGame.Simulation.GetModel<UpgradeDatabaseModel>().GetAvailableUpgradeCount(UpgradeType.Concrete); } catch (Exception) { }
                    SnapshotExtras extras = CaptureExtras(_ipcGame);
                    byte[] bytes = ExportSim(_ipcGame);
                    var sb = new StringBuilder("{\"op\":\"DIAG_FORK\"");
                    sb.Append(",\"nPre\":" + nPre + ",\"horizon\":" + horizon + ",\"bytes\":" + bytes.Length + ",\"prngAtSnapshot\":" + JStr(prng0) + ",\"concreteAtSnapshot\":" + concrete0 + ",\"fingerprintTiles\":" + JStr(fp0.Split(';')[0]));
                    sb.Append(",\"branches\":{");
                    bool first = true;
                    if (obj != null && obj.ContainsKey("branches"))
                    {
                        foreach (object bo in (System.Collections.Generic.List<object>)obj["branches"])
                        {
                            var b = (System.Collections.Generic.Dictionary<string, object>)bo;
                            string name = b.ContainsKey("name") ? (string)b["name"] : ("b" + sb.Length);
                            var edits = new System.Collections.Generic.List<ActionDescriptor>();
                            if (b.ContainsKey("actions")) foreach (object ao in (System.Collections.Generic.List<object>)b["actions"]) edits.Add(ActionIndexToDescriptor((int)(double)ao, _ipcActionBounds));
                            if (!first) sb.Append(",");
                            first = false;
                            sb.Append(RunForkBranch(bytes, extras, name, edits, horizon, cps, interventionSeconds));
                        }
                    }
                    sb.Append("}}");
                    return sb.ToString();
                }
                catch (Exception ex)
                {
                    WriteLine("[Snapshot] EXCEPTION in fork: " + ex);
                    return "{\"op\":\"DIAG_FORK\",\"error\":" + JStr(ex.ToString()) + "}";
                }
            }
