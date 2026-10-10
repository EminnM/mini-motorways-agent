using Factory;
using Motorways.Models;
using Server;

namespace Motorways.Commands;

public class EditTileCommand : Command, IReleasedFromScopeHandler
{
	[Dependency]
	private TilemapModel _tilemap;

	[Dependency]
	private UpgradeDatabaseModel _upgradeDatabase;

	private TileEdit _edit;

	public override void Execute(ISimulation simulation)
	{
		Command.Log.Info("Executing EditTileCommand with {0}.", _edit);
		if (Diagnostics.Verify(_upgradeDatabase.ApplyEdit(_edit, _tilemap), "Failed to apply edit {0} to the upgrade database.", _edit))
		{
			Diagnostics.Verify(_edit.ApplyToTilemap(_tilemap), "Failed to apply edit {0} to the tilemap.", _edit);
		}
		_edit.ApplyToSimulation(simulation);
	}

	public static EditTileCommand Create(IScope scope, TileEdit edit)
	{
		EditTileCommand editTileCommand = scope.Get<EditTileCommand>();
		editTileCommand._edit = edit;
		return editTileCommand;
	}

	public void OnReleasedFromScope(IScope scope)
	{
		if (_edit != null)
		{
			scope.Release(_edit);
			_edit = null;
		}
	}

	public override string ToString()
	{
		return $"[EditTileCommand Edit={_edit}]";
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
