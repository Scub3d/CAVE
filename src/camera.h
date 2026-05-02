#pragma once

#include <GLFW/glfw3.h>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include "common/structs.h"

namespace Cave
{
	enum class CameraMode
	{
		FixedPaceOrbit,  // Framerate-independent orbit (uses delta time)
		FixedRateOrbit,  // Framerate-dependent orbit (constant angular step per frame)
		FreeCamera       // User-controlled with WASD + arrow keys
	};

	class Camera
	{
	private: // Variables
		GLFWwindow* _window;

		CameraMode _cameraMode;

		// Projection parameters
		float _fieldOfView;
		float _aspectRatio;
		float _nearClipPlane;
		float _farClipPlane;

		// Orbit state (shared by both orbit modes)
		float _orbitRadius;
		float _orbitAngle;
		float _orbitElevation;
		float _orbitAngularVelocity;

		// Free camera state
		glm::vec3 _position;
		float _yaw;
		float _pitch;
		float _movementSpeed;
		float _lookSpeed;

	private: // Methods
		void UpdateFixedPaceOrbit(float deltaTimeInSeconds);
		void UpdateFixedRateOrbit();
		void UpdateFreeCamera(float deltaTimeInSeconds);

		glm::vec3 ComputeOrbitPosition() const;
		glm::vec3 ComputeFreeCameraForward() const;
		glm::mat4 BuildViewMatrix() const;
		glm::mat4 BuildProjectionMatrix() const;

	public:
		Camera(GLFWwindow* window, float aspectRatio);
		~Camera() = default;

		Camera(const Camera&) = delete;
		Camera& operator=(const Camera&) = delete;

		void Update(float deltaTimeInSeconds);

		CameraData GetCameraData() const;
		glm::vec3 GetPosition() const;
		glm::vec3 GetForwardDirection() const;

		void SetCameraMode(CameraMode cameraMode);
		CameraMode GetCameraMode() const { return _cameraMode; }

		void SetOrbitRadius(float orbitRadius);
		void SetOrbitAngularVelocity(float orbitAngularVelocity);
		void SetOrbitElevation(float orbitElevation);
		void SetOrbitAngle(float orbitAngle);
		void SetMovementSpeed(float movementSpeed);
		void SetLookSpeed(float lookSpeed);
		void SetFieldOfView(float fieldOfViewInRadians);
		void SetAspectRatio(float aspectRatio);
	};

} // namespace Cave
