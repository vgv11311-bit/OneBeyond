import React, { useEffect, useRef, useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { moveWithinRoom, Obstacle, Vec3 } from './walkthrough';
import { WALK_SETTINGS } from '../../../../scripts/splat-navigation.js';

interface FirstPersonControlsProps {
  moveSpeed?: number;
  lookSpeed?: number;
  initialPosition?: [number, number, number];
  initialYaw?: number;
  initialRotation?: number[][];
  roomBounds?: [number, number, number]; // width, length, height
  roomLimits?: [number, number, number, number]; // minX, maxX, minZ, maxZ
  isMobileTouch?: boolean;
  obstacles?: Obstacle[];
}

export const FirstPersonControls: React.FC<FirstPersonControlsProps> = ({
  moveSpeed = WALK_SETTINGS.speed,
  lookSpeed = WALK_SETTINGS.look,
  initialPosition = [0, 1.4, 1.5],
  initialYaw = 0,
  initialRotation,
  roomBounds = [6.0, 7.0, 3.0],
  roomLimits,
  isMobileTouch = false,
  obstacles = [],
}) => {
  const { camera, gl, performance: renderPerformance } = useThree();
  const keysPressed = useRef<{ [key: string]: boolean }>({});
  const isLocked = useRef(false);
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const velocity = useRef(new THREE.Vector3());
  const forwardVector = useRef(new THREE.Vector3());
  const rightVector = useRef(new THREE.Vector3());
  const moveDirection = useRef(new THREE.Vector3());
  const targetVelocity = useRef(new THREE.Vector3());
  const isDragging = useRef(false);
  const prevMousePos = useRef({ x: 0, y: 0 });
  const initialPoseApplied = useRef(false);

  const [hw, hl, hh] = [roomBounds[0] / 2 - 0.3, roomBounds[1] / 2 - 0.3, roomBounds[2] - 0.2];
  const [minX, maxX, minZ, maxZ] = roomLimits ?? [-hw, hw, -hl, hl];

  useEffect(() => {
    if (!initialPoseApplied.current) {
      camera.position.set(...initialPosition);
      // Apply the reconstructed entrance pose only once. Render metric updates
      // and quality changes must never reset a walking camera.
      const yaw=THREE.MathUtils.degToRad(initialYaw);
      camera.lookAt(initialPosition[0]-Math.sin(yaw), initialPosition[1], initialPosition[2]-Math.cos(yaw));
      if (initialRotation?.length===3 && initialRotation.every(row=>row.length===3 && row.every(Number.isFinite))) {
        const r=initialRotation;
        const matrix=new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],0,
          r[1][0],r[1][1],r[1][2],0,r[2][0],r[2][1],r[2][2],0,0,0,0,1);
        camera.quaternion.setFromRotationMatrix(matrix);
      }
      euler.current.setFromQuaternion(camera.quaternion);
      initialPoseApplied.current = true;
    }

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target;
      if (!isLocked.current && target instanceof HTMLElement && target.closest('input, textarea, select, button, [contenteditable="true"]')) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      keysPressed.current[e.code] = true;
      if (/^(Key[WASDC]|Arrow|Space|Shift|Control)/.test(e.code)) {
        e.preventDefault();
        renderPerformance.regress();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.code] = false;
    };
    const onBlur = () => {
      keysPressed.current = {};
      velocity.current.set(0, 0, 0);
      isDragging.current = false;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement === gl.domElement || isDragging.current) {
        renderPerformance.regress();
        const movementX = e.movementX ?? (e.clientX - prevMousePos.current.x);
        const movementY = e.movementY ?? (e.clientY - prevMousePos.current.y);

        euler.current.y -= movementX * lookSpeed;
        euler.current.x -= movementY * lookSpeed;
        euler.current.x = Math.max(-Math.PI / 2.1, Math.min(Math.PI / 2.1, euler.current.x));

        camera.quaternion.setFromEuler(euler.current);
      }
      prevMousePos.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        gl.domElement.tabIndex = 0;
        gl.domElement.focus({ preventScroll: true });
        isDragging.current = true;
        prevMousePos.current = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging.current = false;
    };

    const onPointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
      if (isLocked.current) {
        gl.domElement.tabIndex = 0;
        gl.domElement.focus({ preventScroll: true });
      }
      if (!isLocked.current) onBlur();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('blur', onBlur);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('mousemove', onMouseMove);
    gl.domElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    document.addEventListener('pointerlockchange', onPointerLockChange);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('blur', onBlur);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('mousemove', onMouseMove);
      gl.domElement.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
    };
  }, [camera, gl.domElement, initialPosition[0], initialPosition[1], initialPosition[2], initialYaw, initialRotation, lookSpeed, renderPerformance]);

  useFrame((_, delta) => {
    delta = Math.min(delta, 0.05);
    const forward = forwardVector.current.set(0, 0, -1).applyQuaternion(camera.quaternion);
    forward.y = 0;
    forward.normalize();

    const right = rightVector.current.set(1, 0, 0).applyQuaternion(camera.quaternion);
    right.y = 0;
    right.normalize();

    const direction = moveDirection.current.set(0, 0, 0);

    // Check WASD and Arrow Keys
    const keys = keysPressed.current;
    if (keys['KeyW'] || keys['ArrowUp']) direction.add(forward);
    if (keys['KeyS'] || keys['ArrowDown']) direction.sub(forward);
    if (keys['KeyD'] || keys['ArrowRight']) direction.add(right);
    if (keys['KeyA'] || keys['ArrowLeft']) direction.sub(right);

    // Height Controls: Space (Up), KeyC / Shift (Down)
    if (keys['Space']) direction.y += 1.0;
    if (keys['KeyC'] || keys['ControlLeft']) direction.y -= 1.0;

    const sprint = keys['ShiftLeft'] || keys['ShiftRight'] ? WALK_SETTINGS.sprint : 1.0;

    // Smoothly approach a capped target velocity. The previous acceleration /
    // friction loop could overshoot at uneven frame rates, which felt like a
    // collision or stutter while moving through the furnished sample.
    const target = targetVelocity.current.set(0, 0, 0);
    if (direction.lengthSq() > 0) {
      // Keep the adaptive DPR in its fast state for the entire walk/sprint,
      // then let it recover to full quality shortly after movement stops.
      renderPerformance.regress();
      target.copy(direction.normalize()).multiplyScalar(moveSpeed * sprint);
    }
    velocity.current.lerp(target, 1 - Math.exp(-WALK_SETTINGS.damping * delta));

    // Update position with room boundary containment
    camera.position.fromArray(moveWithinRoom(camera.position.toArray() as Vec3,
      [velocity.current.x * delta, velocity.current.y * delta, velocity.current.z * delta],
      [minX,maxX,minZ,maxZ],roomBounds[2],obstacles));
    gl.domElement.dataset.cameraPosition = camera.position
      .toArray()
      .map((value) => value.toFixed(3))
      .join(',');
  });

  return null;
};
