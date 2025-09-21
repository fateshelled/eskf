#include <GL/glut.h>

#include <random>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

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
// At the top of generateSimulatedData
    std::mt19937 gen(seed);

    const double pos_noise_level = 0.2;
    const double ori_noise_level = 0.05;
    std::uniform_real_distribution<> distrib_pos(-pos_noise_level, pos_noise_level);
    std::uniform_real_distribution<> distrib_ori(0.0, ori_noise_level);
    std::uniform_real_distribution<> distrib_axis(-1.0, 1.0);

    std::vector<Eigen::Vector3d> true_positions;
    std::vector<Eigen::Quaterniond> true_orientations;
    std::vector<double> timestamps;

    const int num_steps = 100;
    const double dt = 0.1;
    true_positions.reserve(num_steps);
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

    ErrorStateKF eskf;
    Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
    initial_pose.translation() = Eigen::Vector3d::Zero();
    initial_pose.linear() = Eigen::Quaterniond::Identity().toRotationMatrix();

    eskf.setInitialPose(initial_pose);
    eskf.setMeasurementNoise(1.0, 0.5);
    eskf.setProcessNoise(0.2, 1.5, 0.2, 5.0);

    for (int i = 0; i < num_steps; ++i)
    {
        Eigen::Isometry3d noisy_pose = Eigen::Isometry3d::Identity();
        noisy_pose.translation() = noisy_positions[i];
        noisy_pose.linear() = noisy_orientations[i].toRotationMatrix();

        eskf.update(noisy_pose, timestamps[i]);

        const Eigen::Isometry3d pose = eskf.getCurrentPose();
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

    glutDisplayFunc([]() { TrajectoryVisualizer::getInstance()->render(); });
    glutKeyboardFunc([](unsigned char key, int x, int y) { TrajectoryVisualizer::getInstance()->handleKeyboard(key, x, y); });
    glutSpecialFunc([](int key, int x, int y) { TrajectoryVisualizer::getInstance()->handleSpecialKeys(key, x, y); });
    glutReshapeFunc([](int w, int h) { TrajectoryVisualizer::getInstance()->handleReshape(w, h); });
    glutMouseFunc([](int button, int state, int x, int y) { TrajectoryVisualizer::getInstance()->handleMouse(button, state, x, y); });
    glutMotionFunc([](int x, int y) { TrajectoryVisualizer::getInstance()->handleMouseMotion(x, y); });

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
