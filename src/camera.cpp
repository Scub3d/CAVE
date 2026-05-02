#include "camera.h"

namespace Cave
{

	Camera::Camera(GLFWwindow* window, float aspectRatio)
		: _window{window},
		  _cameraMode{CameraMode::FixedPaceOrbit},
		  _fieldOfView{glm::radians(45.0f)},
		  _aspectRatio{aspectRatio},
		  _nearClipPlane{0.1f},
		  _farClipPlane{10000.0f},
		  _orbitRadius{50.0f},
		  _orbitAngle{0.0f},
		  _orbitElevation{0.4f},
		  _orbitAngularVelocity{0.03f},
		  _position{0.0f, 0.0f, 50.0f},
		  _yaw{glm::pi<float>()},
		  _pitch{0.0f},
		  _movementSpeed{20.0f},
		  _lookSpeed{1.5f}
	{
	}

	void Camera::Update(float deltaTimeInSeconds)
	{
		switch (_cameraMode)
		{
		case CameraMode::FixedPaceOrbit:
			UpdateFixedPaceOrbit(deltaTimeInSeconds);
			break;
		case CameraMode::FixedRateOrbit:
			UpdateFixedRateOrbit();
			break;
		case CameraMode::FreeCamera:
			UpdateFreeCamera(deltaTimeInSeconds);
			break;
		}
	}

	void Camera::UpdateFixedPaceOrbit(float deltaTimeInSeconds)
	{
		_orbitAngle += _orbitAngularVelocity * deltaTimeInSeconds;
		if (_orbitAngle > glm::two_pi<float>())
			_orbitAngle -= glm::two_pi<float>();
	}

	void Camera::UpdateFixedRateOrbit()
	{
		_orbitAngle += _orbitAngularVelocity;
		if (_orbitAngle > glm::two_pi<float>())
			_orbitAngle -= glm::two_pi<float>();
	}

	void Camera::UpdateFreeCamera(float deltaTimeInSeconds)
	{
		// Look direction from arrow keys
		if (glfwGetKey(_window, GLFW_KEY_LEFT) == GLFW_PRESS)
			_yaw -= _lookSpeed * deltaTimeInSeconds;
		if (glfwGetKey(_window, GLFW_KEY_RIGHT) == GLFW_PRESS)
			_yaw += _lookSpeed * deltaTimeInSeconds;
		if (glfwGetKey(_window, GLFW_KEY_UP) == GLFW_PRESS)
			_pitch += _lookSpeed * deltaTimeInSeconds;
		if (glfwGetKey(_window, GLFW_KEY_DOWN) == GLFW_PRESS)
			_pitch -= _lookSpeed * deltaTimeInSeconds;

		// Clamp pitch to avoid gimbal lock
		_pitch = glm::clamp(_pitch, glm::radians(-89.0f), glm::radians(89.0f));

		glm::vec3 forward = ComputeFreeCameraForward();
		glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));

		// WASD movement
		float speed = _movementSpeed * deltaTimeInSeconds;
		if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
			_position += forward * speed;
		if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
			_position -= forward * speed;
		if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
			_position -= right * speed;
		if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
			_position += right * speed;
	}

	glm::vec3 Camera::ComputeOrbitPosition() const
	{
		float x = _orbitRadius * glm::cos(_orbitElevation) * glm::sin(_orbitAngle);
		float y = _orbitRadius * glm::sin(_orbitElevation);
		float z = _orbitRadius * glm::cos(_orbitElevation) * glm::cos(_orbitAngle);
		return glm::vec3(x, y, z);
	}

	glm::vec3 Camera::ComputeFreeCameraForward() const
	{
		glm::vec3 forward;
		forward.x = glm::cos(_pitch) * glm::sin(_yaw);
		forward.y = glm::sin(_pitch);
		forward.z = glm::cos(_pitch) * glm::cos(_yaw);
		return glm::normalize(forward);
	}

	glm::mat4 Camera::BuildViewMatrix() const
	{
		glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

		switch (_cameraMode)
		{
		case CameraMode::FixedPaceOrbit:
		case CameraMode::FixedRateOrbit:
		{
			glm::vec3 orbitPosition = ComputeOrbitPosition();
			return glm::lookAt(orbitPosition, glm::vec3(0.0f), worldUp);
		}
		case CameraMode::FreeCamera:
		{
			glm::vec3 forward = ComputeFreeCameraForward();
			return glm::lookAt(_position, _position + forward, worldUp);
		}
		}

		return glm::mat4(1.0f);
	}

	glm::mat4 Camera::BuildProjectionMatrix() const
	{
		glm::mat4 projection = glm::perspective(_fieldOfView, _aspectRatio, _nearClipPlane, _farClipPlane);
		return projection;
	}

	CameraData Camera::GetCameraData() const
	{
		CameraData cameraData{};
		cameraData.model = glm::mat4(1.0f);
		cameraData.view = BuildViewMatrix();
		cameraData.projection = BuildProjectionMatrix();
		return cameraData;
	}

	glm::vec3 Camera::GetPosition() const
	{
		switch (_cameraMode)
		{
		case CameraMode::FixedPaceOrbit:
		case CameraMode::FixedRateOrbit:
			return ComputeOrbitPosition();
		case CameraMode::FreeCamera:
			return _position;
		}
		return glm::vec3(0.0f);
	}

	glm::vec3 Camera::GetForwardDirection() const
	{
		if (_cameraMode == CameraMode::FreeCamera)
			return ComputeFreeCameraForward();

		// Orbit modes always look at the world origin.
		glm::vec3 position = ComputeOrbitPosition();
		float lengthSquared = glm::dot(position, position);
		if (lengthSquared < 1e-8f)
			return glm::vec3(0.0f, 0.0f, -1.0f);
		return glm::normalize(-position);
	}

	void Camera::SetCameraMode(CameraMode cameraMode)
	{
		if (_cameraMode == cameraMode)
			return;

		// When transitioning to FreeCamera, initialize position/orientation from current orbit
		if (cameraMode == CameraMode::FreeCamera)
		{
			_position = ComputeOrbitPosition();
			_yaw = _orbitAngle + glm::pi<float>();
			_pitch = -_orbitElevation;
		}

		// When transitioning from FreeCamera to an orbit mode, compute orbit state from position
		if (_cameraMode == CameraMode::FreeCamera)
		{
			_orbitRadius = glm::length(_position);
			_orbitAngle = glm::atan(_position.x, _position.z);
			_orbitElevation = glm::asin(glm::clamp(_position.y / _orbitRadius, -1.0f, 1.0f));
		}

		_cameraMode = cameraMode;
	}

	void Camera::SetOrbitRadius(float orbitRadius) { _orbitRadius = orbitRadius; }
	void Camera::SetOrbitAngularVelocity(float orbitAngularVelocity) { _orbitAngularVelocity = orbitAngularVelocity; }
	void Camera::SetOrbitElevation(float orbitElevation) { _orbitElevation = orbitElevation; }
	void Camera::SetOrbitAngle(float orbitAngle) { _orbitAngle = orbitAngle; }
	void Camera::SetMovementSpeed(float movementSpeed) { _movementSpeed = movementSpeed; }
	void Camera::SetLookSpeed(float lookSpeed) { _lookSpeed = lookSpeed; }
	void Camera::SetFieldOfView(float fieldOfViewInRadians) { _fieldOfView = fieldOfViewInRadians; }
	void Camera::SetAspectRatio(float aspectRatio) { _aspectRatio = aspectRatio; }

} // namespace Cave
