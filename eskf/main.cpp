#include <GL/glut.h>

#include <random>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "imu_sr_eskf.hpp"
#include "eskf.hpp"
#include "trajectory_visualizer.hpp"

namespace
{
    void printUsage(const char *program_name)
    {
        std::cout << "Usage:\n";
        std::cout << "  " << program_name << "            # Generate simulated data and visualize\n";
        std::cout << "  " << program_name << " --load <raw.tum> <filtered.tum> <gt.tum>\n";
        std::cout << "      Load trajectories from existing TUM files and visualize\n";
    }

    void generateSimulatedData(TrajectoryData &trajectory_data)
    {
        trajectory_data.clear();

        const int seed = 12345;
        std::mt19937 gen(seed);

        const double pos_noise_level = 0.2;
        const double ori_noise_level = 0.05;
        std::uniform_real_distribution<> distrib_pos(-pos_noise_level, pos_noise_level);
        std::uniform_real_distribution<> distrib_ori(0.0, ori_noise_level);
        std::uniform_real_distribution<> distrib_axis(-1.0, 1.0);

        std::vector<Eigen::Vector3d> true_positions;
        std::vector<Eigen::Vector3d> true_velocities;
        std::vector<Eigen::Vector3d> true_accelerations;
        std::vector<Eigen::Quaterniond> true_orientations;
        std::vector<double> timestamps;

        const int num_steps = 100;
        const double dt = 0.1;
        const double omega = (2.0 * M_PI / static_cast<double>(num_steps)) / dt;
        true_positions.reserve(num_steps);
        true_velocities.reserve(num_steps);
        true_accelerations.reserve(num_steps);
        true_orientations.reserve(num_steps);
        timestamps.reserve(num_steps);

        for (int i = 0; i < num_steps; ++i)
        {
            const double t = static_cast<double>(i) * 2.0 * M_PI / static_cast<double>(num_steps);

            const Eigen::Vector3d true_pos(4.0 * std::sin(t), 0.5 * std::sin(2.0 * t), 2.0 * std::sin(2.0 * t));
            Eigen::Vector3d direction;
            if (i > 0)
            {
                direction = (true_pos - true_positions.back()).normalized();
            }
            else
            {
                direction = Eigen::Vector3d::UnitX();
            }

            Eigen::Quaterniond true_ori = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), direction);

            true_positions.push_back(true_pos);
            true_orientations.push_back(true_ori);
            timestamps.push_back(static_cast<double>(i) * dt);

            trajectory_data.ground_truth.emplace_back(timestamps.back(), true_pos, true_ori, false);
        }

        true_velocities.resize(num_steps);
        true_accelerations.resize(num_steps);
        for (int i = 0; i < num_steps; ++i)
        {
            const double t_param = static_cast<double>(i) * 2.0 * M_PI / static_cast<double>(num_steps);

            true_velocities[i] = Eigen::Vector3d(
                4.0 * omega * std::cos(t_param),
                omega * std::cos(2.0 * t_param),
                4.0 * omega * std::cos(2.0 * t_param));

            true_accelerations[i] = Eigen::Vector3d(
                -4.0 * omega * omega * std::sin(t_param),
                -2.0 * omega * omega * std::sin(2.0 * t_param),
                -8.0 * omega * omega * std::sin(2.0 * t_param));
        }

        std::vector<Eigen::Vector3d> noisy_positions;
        std::vector<Eigen::Quaterniond> noisy_orientations;
        noisy_positions.reserve(num_steps);
        noisy_orientations.reserve(num_steps);

        for (int i = 0; i < num_steps; ++i)
        {
            const Eigen::Vector3d noise_pos(distrib_pos(gen), distrib_pos(gen), distrib_pos(gen));
            const Eigen::Vector3d noisy_pos = true_positions[i] + noise_pos;

            const Eigen::Vector3d noise_axis = Eigen::Vector3d(distrib_axis(gen), distrib_axis(gen), distrib_axis(gen)).normalized();
            const double noise_angle = distrib_ori(gen);
            const Eigen::Quaterniond noise_quat(Eigen::AngleAxisd(noise_angle, noise_axis));
            const Eigen::Quaterniond noisy_ori = true_orientations[i] * noise_quat;

            noisy_positions.push_back(noisy_pos);
            noisy_orientations.push_back(noisy_ori);

            trajectory_data.raw_trajectory.emplace_back(timestamps[i], noisy_pos, noisy_ori, false);
        }

        const Eigen::Vector3d gravity(0.0, 0.0, -9.81);
        std::normal_distribution<double> gyro_noise(0.0, 0.002);
        std::normal_distribution<double> accel_noise(0.0, 0.05);

        std::vector<Eigen::Vector3d> gyro_measurements;
        std::vector<Eigen::Vector3d> accel_measurements;
        gyro_measurements.reserve(num_steps - 1);
        accel_measurements.reserve(num_steps - 1);

        for (int i = 1; i < num_steps; ++i)
        {
            const double dt_i = timestamps[i] - timestamps[i - 1];
            const Eigen::Quaterniond dq = true_orientations[i - 1].conjugate() * true_orientations[i];
            Eigen::Vector3d gyro = IMU_SR_ESKF::logSO3(dq) / dt_i;
            Eigen::Vector3d acc_world = true_accelerations[i];
            const Eigen::Matrix3d Rwb_prev = true_orientations[i - 1].toRotationMatrix();
            Eigen::Vector3d specific_force = Rwb_prev.transpose() * (acc_world - gravity);

            for (int axis = 0; axis < 3; ++axis)
            {
                gyro(axis) += gyro_noise(gen);
                specific_force(axis) += accel_noise(gen);
            }

            gyro_measurements.push_back(gyro);
            accel_measurements.push_back(specific_force);
        }

        IMU_SR_ESKF eskf;
        eskf.setState(noisy_positions.front(), true_velocities.front(), noisy_orientations.front(),
                      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        eskf.setMeasurementNoiseStandardDeviations(0.1, 0.1);
        eskf.setProcessNoiseStandardDeviations(0.01, 0.1, 1e-4, 1e-4);

        for (int i = 0; i < num_steps; ++i)
        {
            if (i > 0)
            {
                const double dt_i = timestamps[i] - timestamps[i - 1];
                eskf.predict(dt_i, gyro_measurements[i - 1], accel_measurements[i - 1], gravity);
            }
            eskf.update(noisy_positions[i], noisy_orientations[i]);

            const Eigen::Isometry3d pose = eskf.getPose();
            trajectory_data.filtered_trajectory.emplace_back(timestamps[i], pose.translation(),
                                                             Eigen::Quaterniond(pose.linear()), true);
        }
    }
} // namespace

int main(int argc, char **argv)
{
    glutInit(&argc, argv);

    TrajectoryData trajectory_data;
    bool loaded_from_files = false;

    if (argc > 1)
    {
        const std::string option = argv[1];
        if (option == "--load")
        {
            if (argc < 5)
            {
                std::cerr << "Error: --load requires three file paths" << std::endl;
                printUsage(argv[0]);
                return 1;
            }

            if (!trajectory_data.loadTumFiles(argv[2], argv[3], argv[4]))
            {
                std::cerr << "Error: failed to load one or more TUM files" << std::endl;
                return 1;
            }
            loaded_from_files = true;
        }
        else if (option == "--help" || option == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown option: " << option << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!loaded_from_files)
    {
        generateSimulatedData(trajectory_data);
        if (!trajectory_data.saveTumFiles("trajectory"))
        {
            std::cerr << "Warning: Failed to save simulated trajectories to TUM files" << std::endl;
        }
    }

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("ESKF Trajectory Visualization");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);

    TrajectoryVisualizer visualizer(&trajectory_data);
    TrajectoryVisualizer::setInstance(&visualizer);

    glutDisplayFunc([]()
                    { TrajectoryVisualizer::getInstance()->render(); });
    glutKeyboardFunc([](unsigned char key, int x, int y)
                     { TrajectoryVisualizer::getInstance()->handleKeyboard(key, x, y); });
    glutSpecialFunc([](int key, int x, int y)
                    { TrajectoryVisualizer::getInstance()->handleSpecialKeys(key, x, y); });
    glutReshapeFunc([](int w, int h)
                    { TrajectoryVisualizer::getInstance()->handleReshape(w, h); });
    glutMouseFunc([](int button, int state, int x, int y)
                  { TrajectoryVisualizer::getInstance()->handleMouse(button, state, x, y); });
    glutMotionFunc([](int x, int y)
                   { TrajectoryVisualizer::getInstance()->handleMouseMotion(x, y); });

    std::cout << "ESKF Trajectory Visualization Controls:\n";
    std::cout << "  r - Toggle raw trajectory\n";
    std::cout << "  f - Toggle filtered trajectory\n";
    std::cout << "  t - Toggle ground truth trajectory\n";
    std::cout << "  c - Toggle coordinate frames\n";
    std::cout << "  g - Toggle ground plane grid\n";
    std::cout << "  Left/Right Arrows - Previous/Next frame\n";
    std::cout << "  Up/Down Arrows - Adjust camera angle\n";
    std::cout << "  Mouse Drag - Rotate camera\n";
    std::cout << "  Shift+Mouse Drag or Right Mouse - Pan camera\n";
    std::cout << "  +/- - Zoom in/out\n";
    std::cout << "  a - Start/stop animation\n";
    std::cout << "  1/2 - Decrease/Increase animation speed\n";
    std::cout << "  s - Save trajectory data to TUM files\n";
    std::cout << "  q/ESC - Quit\n";

    glutMainLoop();

    return 0;
}
