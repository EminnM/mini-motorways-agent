using Client;
using Factory;
using FixMath;
using Motorways.Audio;
using Motorways.Models;
using Motorways.UI;
using Motorways.Views;
using UnityEngine;

namespace Motorways.Actions;

public class DrawRoadAction : MotorwaysPlayerAction
{
	[Dependency]
	private IAudioSystem _audioSystem;

	[Dependency]
	private IScope _scope;

	[Dependency]
	private ViewClient _viewClient;

	[Dependency]
	private CameraView _cameraView;

	[Dependency]
	private NotificationView _notificationView;

	[Dependency]
	private VisualConstantsData _constants;

	[Dependency]
	private ActivePlayer _player;

	private static readonly float OriginalStepMultiplier = 1.65f;

	private Vector2Int _currentCoordinates;

	public static readonly Fix64 UTurnNubChangeTolerance = (Fix64)0.2f;

	private Vector2 _previousMousePosition;

	private NewRoadPreview _roadPreview;

	private Vector2Int _lastErrorPosition;

	private bool _currentlyInErrorState;

	protected override bool ManuallyHandlesReservations => true;

	public override bool PreventsCursorAcceleration => true;

	public override void OnActionBegin(float timestamp)
	{
		base.OnActionBegin(timestamp);
		SetColourWidgetRadialVisible(visible: false);
		if (_inputState.GetButtonDown(25) || (base.OwningGroup.InstigatingInputEvent.Source == InputEventSource.Touch && !_cameraView.IsFocussedIn) || _gameUI.CurrentRoadDrawMode != RoadDrawMode.Add)
		{
			OnActionCancel();
			return;
		}
		_currentCoordinates = GetPointerTilePosition();
		_previousMousePosition = GetPointerWorldPosition();
		Tile orCreateTile = _tilemapView.GetOrCreateTile(_currentCoordinates);
		if (orCreateTile != null)
		{
			TileDirectionBitfield.Enumerator enumerator = orCreateTile.GetMotorwayRamps(RoadState.Planned | RoadState.Active).GetEnumerator();
			while (enumerator.MoveNext())
			{
				TileDirection current = enumerator.Current;
				if (!_tilemapView.GetMotorway(orCreateTile.GetMotorwayInDirection(current, RoadState.Planned | RoadState.Active)).IsPermanent)
				{
					OnActionCancel();
					return;
				}
			}
		}
		UpdateCursorPosition();
		SetCursorVisible(visible: true);
		if (_gameUI.IsFocusPointActive)
		{
			_gameUI.SetFocusPointPosition(_tilemapView.GetScreenPositionFromTileCoordinates(_currentCoordinates));
		}
		PlayerAction.Log.Info("Beginning DrawRoadAction from tile coordinates {0}.", _currentCoordinates);
	}

	public override void Tick(float frameTime)
	{
		base.Tick(frameTime);
		if (base.ActionState != State.Begun)
		{
			return;
		}
		if (base.Scope.Get<EditMenuPanel>().IsOpen)
		{
			_gameUI.ConfirmEditMenuEdit();
			OnActionCancel();
			return;
		}
		Vector2Int pointerTilePosition = GetPointerTilePosition();
		Fix64 expansionTime = _clockModel.ExpansionTime;
		bool flag = _city.IsTileInPlayableArea(_currentCoordinates, expansionTime);
		if (pointerTilePosition != _currentCoordinates)
		{
			if (flag && _city.IsTileInPlayableArea(pointerTilePosition, expansionTime))
			{
				Vector2 pointerWorldPosition = GetPointerWorldPosition();
				Vector2 vector = TilemapView.GetWorldPositionForCoordinates(_currentCoordinates);
				Vector2 vector2 = pointerWorldPosition - vector;
				TileDirection closestDirection = TileUtilities.GetClosestDirection(vector2.normalized);
				float num = _constants.RoadDrawingStepDistance;
				if (TileUtilities.IsDirectionDiagonal(closestDirection))
				{
					num = _constants.DiagonalRoadDrawingStepDistance;
				}
				if (FeatureToggle.IsFeatureDisabled(Feature.RoadDrawingAnimations))
				{
					num = OriginalStepMultiplier;
				}
				float num2 = num * (TileUtilities.IsDirectionDiagonal(closestDirection) ? ((float)TilemapModel.HalfTileWidth * Mathf.Sqrt(2f)) : ((float)TilemapModel.HalfTileWidth));
				while (vector2.sqrMagnitude >= num2 * num2)
				{
					Vector2Int adjacentCoordinates = TileUtilities.GetAdjacentCoordinates(_currentCoordinates, closestDirection);
					if (!_city.IsTileInPlayableArea(adjacentCoordinates, expansionTime))
					{
						break;
					}
					PlayerAction.Log.Info("Building from {0} in direction {1}.", _currentCoordinates, vector2);
					if (!TryAddRoadInDirection(ref _currentCoordinates, vector2))
					{
						break;
					}
					vector = TilemapView.GetWorldPositionForCoordinates(_currentCoordinates);
					vector2 = pointerWorldPosition - vector;
					closestDirection = TileUtilities.GetClosestDirection(vector2.normalized);
					num2 = num * (TileUtilities.IsDirectionDiagonal(closestDirection) ? ((float)TilemapModel.HalfTileWidth * Mathf.Sqrt(2f)) : ((float)TilemapModel.HalfTileWidth));
				}
			}
			else
			{
				_currentCoordinates = pointerTilePosition;
				_previousMousePosition = GetPointerWorldPosition();
			}
		}
		if (flag && !_city.Definition.TileIsUnderAMountain(_currentCoordinates))
		{
			if (FeatureToggle.IsFeatureEnabled(Feature.RoadDrawingAnimations))
			{
				UpdateRoadPreview();
			}
			else
			{
				HideRoadPreview();
			}
		}
		else
		{
			HideRoadPreview();
		}
		UpdateCursorPosition();
	}

	private void HideRoadPreview()
	{
		if (_roadPreview != null)
		{
			_roadPreview.Remove();
			_roadPreview = null;
		}
	}

	private void UpdateRoadPreview()
	{
		if (_roadPreview == null)
		{
			_roadPreview = _scope.Get<NewRoadPreview>();
			_viewClient.AddView(_roadPreview);
			CheckHazardStripes();
		}
		if (!_currentlyInErrorState)
		{
			CheckHazardStripes();
		}
		Vector2Int fromCoordinates = (_currentlyInErrorState ? _lastErrorPosition : _currentCoordinates);
		Vector2 pointerWorldPosition = GetPointerWorldPosition();
		_roadPreview.SetPosition(fromCoordinates, pointerWorldPosition);
	}

	private bool IsCurrentTileHouse()
	{
		Tile tile = _tilemapView.GetTile(_currentCoordinates);
		if (tile != null)
		{
			return tile.ContentType == TileContentType.House;
		}
		return false;
	}

	private void CheckHazardStripes()
	{
		bool stripesEnabled = !IsCurrentTileHouse() && !_upgradeDatabase.HasUpgradeAvailable(UpgradeType.Concrete);
		_roadPreview.SetHazardStripesEnabled(stripesEnabled, tween: true);
	}

	private bool TryAddRoadInDirection(ref Vector2Int currentPosition, Vector2 directionVector)
	{
		TileDirection closestDirection = TileUtilities.GetClosestDirection(directionVector);
		TileEditResult tileEditResult = _tileEditor.AddRoad(_tilemapView, currentPosition, closestDirection);
		if (tileEditResult.edit != null || !tileEditResult.IsSuccessful)
		{
			AudioEventType type = AudioEventType.BuildRoad;
			Vector2Int adjacentCoordinates = TileUtilities.GetAdjacentCoordinates(currentPosition, closestDirection);
			if (_city.Definition.TileIsOverWater(adjacentCoordinates))
			{
				type = AudioEventType.BuildBridge;
			}
			else if (_city.Definition.TileIsUnderAMountain(adjacentCoordinates))
			{
				type = AudioEventType.BuildTunnel;
			}
			_audioSystem.ScheduleEvent(AudioEvent.CreateEvent(-1.0, type, GetPan().x, -1f, tileEditResult.IsSuccessful));
			if (!tileEditResult.IsSuccessful && !_currentlyInErrorState)
			{
				_lastErrorPosition = tileEditResult.errorPosition;
			}
		}
		if (tileEditResult.IsSuccessful)
		{
			AddTileEdit(tileEditResult.edit, EditExecuteTiming.Immediate);
			_notificationView.HideNotification();
			_notificationView.HideAlertIcon();
			_feedbackGenerator.GenerateFeedback(HapticFeedbackType.LightImpact);
			currentPosition = TileUtilities.GetAdjacentCoordinates(currentPosition, closestDirection);
			_previousMousePosition = GetPointerWorldPosition();
			_currentlyInErrorState = false;
			if (_roadPreview != null)
			{
				_roadPreview.SetHazardStripesEnabled(stripesEnabled: false, tween: true);
			}
			return true;
		}
		if (tileEditResult.resultCode == TileEditResultCode.EditAlreadyExists)
		{
			_notificationView.HideNotification();
			_notificationView.HideAlertIcon();
			currentPosition = TileUtilities.GetAdjacentCoordinates(currentPosition, closestDirection);
			_previousMousePosition = GetPointerWorldPosition();
		}
		else
		{
			if (!_currentlyInErrorState)
			{
				_notificationView.AddNotification(tileEditResult.resultCode, tileEditResult.errorPosition);
				_currentlyInErrorState = true;
				if (_roadPreview != null)
				{
					_roadPreview.SetHazardStripesEnabled(stripesEnabled: true, tween: true);
				}
			}
			_feedbackGenerator.GenerateFeedback(HapticFeedbackType.MediumImpact);
			currentPosition = TileUtilities.GetAdjacentCoordinates(currentPosition, closestDirection);
			_previousMousePosition = GetPointerWorldPosition();
		}
		return false;
	}

	protected override void SetCursorVisible(bool visible)
	{
		_gameUI.SetRoadCursorActive(visible);
		if (!_cameraView.IsFocussedIn)
		{
			SetWorldGridVisible(visible);
			_tilemapView.viewMode = (visible ? TilemapView.ViewMode.Edit : TilemapView.ViewMode.Normal);
		}
		HideRoadPreview();
	}

	public override void ObserveInput(float timestamp, InputEvent inputEvent, bool overUI)
	{
		if (inputEvent.Source == InputEventSource.Generic && _player.IsTapDrawEnabled && inputEvent.ButtonState == InputEventButtonState.JustUp)
		{
			OnActionBegin(timestamp);
		}
		else
		{
			OnActionComplete();
		}
	}

	public override void OnActionComplete()
	{
		PlayerAction.Log.Info("Completing DrawRoadAction.");
		if (FeatureToggle.IsFeatureEnabled(Feature.RoadDrawingEndTileCommit))
		{
			Vector2Int pointerTilePosition = GetPointerTilePosition();
			Vector2Int vector2Int = pointerTilePosition - _currentCoordinates;
			Fix64 expansionTime = _clockModel.ExpansionTime;
			if (pointerTilePosition != _currentCoordinates && vector2Int.magnitude <= Vector2Int.one.magnitude && _city.IsTileInPlayableArea(_currentCoordinates, expansionTime) && _city.IsTileInPlayableArea(pointerTilePosition, expansionTime))
			{
				TryAddRoadInDirection(ref _currentCoordinates, vector2Int);
			}
		}
		SetCursorVisible(visible: false);
		_notificationView.HideAlertIcon();
		_notificationView.CancelNotification();
		base.OnActionComplete();
	}

	public override void OnActionCancel()
	{
		PlayerAction.Log.Info("Cancelling DrawRoadAction.");
		base.OnActionCancel();
		SetCursorVisible(visible: false);
		_notificationView.CancelNotification();
		_notificationView.HideAlertIcon();
	}

	public override void Reset()
	{
		base.Reset();
		_currentCoordinates = default(Vector2Int);
		_previousMousePosition = default(Vector2);
		_currentlyInErrorState = false;
		_lastErrorPosition = default(Vector2Int);
	}

	public static MotorwaysPlayerAction Create(PlayerActionGroup owningGroup, IScope scope, float timestamp)
	{
		InputState inputState = scope.Get<InputState>();
		if (inputState.GetButtonDown(25) || inputState.GetButton(25) || scope.Get<GameUIScreen>().CurrentRoadDrawMode != RoadDrawMode.Add)
		{
			return DragClearTileAction.Create(owningGroup, scope, timestamp);
		}
		DrawRoadAction drawRoadAction = scope.Get<DrawRoadAction>();
		drawRoadAction.InitializeAction(owningGroup, timestamp);
		drawRoadAction._playerPositionSource = (MotorwaysPlayerAction.DoesInputTypeUseFocusPoint(owningGroup.InstigatingInputEvent.Source) ? PlayerPositionSource.FocusPoint : PlayerPositionSource.InputEvent);
		int num;
		if (owningGroup.InstigatingInputEvent.Source == InputEventSource.Generic)
		{
			num = (scope.Get<ActivePlayer>().IsTapDrawEnabled ? 1 : 0);
			if (num != 0)
			{
				drawRoadAction.RegisterObserveInputEvent(InputEventFilter.CreateEventFilter(InputEventSource.Generic, owningGroup.InstigatingInputEvent.InputAction, InputEventButtonState.JustDown), ObserverGreediness.BlocksNewActions);
			}
		}
		else
		{
			num = 0;
		}
		drawRoadAction.RegisterObserveInputEvent(InputEventFilter.CreateEventFilter(owningGroup.InstigatingInputEvent.Source, owningGroup.InstigatingInputEvent.InputAction, InputEventButtonState.JustUp), ObserverGreediness.BlocksNewActions);
		drawRoadAction.RegisterObserveInputEvent(InputEventFilter.CreateEventFilter(owningGroup.InstigatingInputEvent.Source, 18, InputEventButtonState.JustDown), ObserverGreediness.BlocksNewActions);
		drawRoadAction.RegisterObserveInputEvent(InputEventFilter.CreateMouseEventFilter(20, InputEventButtonState.JustDown), ObserverGreediness.BlocksNewActions);
		if (owningGroup.InstigatingInputEvent.Source == InputEventSource.Touch)
		{
			drawRoadAction.RegisterObserveInputEvent(InputEventFilter.CreateTouchEventFilter(0, InputEventButtonState.JustUp), ObserverGreediness.BlocksNewActions);
		}
		if (num == 0)
		{
			drawRoadAction.OnActionBegin(timestamp);
		}
		return drawRoadAction;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
