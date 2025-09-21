#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <GL/glu.h>
#include <GL/glut.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TrajectoryData
{
public:
    struct PoseData
    {
        double timestamp;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        bool is_filtered;

        PoseData(double time, const Eigen::Vector3d &pos, const Eigen::Quaterniond &ori, bool filtered = false)
            : timestamp(time), position(pos), orientation(ori), is_filtered(filtered) {}
    };

    std::vector<PoseData> raw_trajectory;
    std::vector<PoseData> filtered_trajectory;
    std::vector<PoseData> ground_truth;
    mutable std::mutex data_mutex;

    void clear();

    bool saveTumFiles(const std::string &prefix) const;
    bool loadTumFiles(const std::string &raw_path,
                      const std::string &filtered_path,
                      const std::string &ground_truth_path);

private:
    bool saveTumFile(const std::string &filename, const std::vector<PoseData> &trajectory) const;
    bool loadTumFile(const std::string &filename,
                     std::vector<PoseData> &trajectory,
                     bool is_filtered);
};

class TrajectoryVisualizer
{
public:
    explicit TrajectoryVisualizer(TrajectoryData *data);

    void render();
    void handleKeyboard(unsigned char key, int x, int y);
    void handleSpecialKeys(int key, int x, int y);
    void handleReshape(int width, int height);
    void handleMouse(int button, int state, int x, int y);
    void handleMouseMotion(int x, int y);

    void advanceFrame();

    static void setInstance(TrajectoryVisualizer *vis);
    static TrajectoryVisualizer *getInstance();
    static void timerCallback(int);

private:
    void drawCoordinateFrame(float scale) const;
    void drawGrid(int size, float step) const;
    void drawRobot(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                   float size, bool is_filtered) const;
    void drawTrajectory(const std::vector<TrajectoryData::PoseData> &trajectory,
                        int trajectory_type) const;
    void drawDirectionArrow(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                            float size, const float color[3], bool show_xyz = false) const;
    void drawEulerAngles(const Eigen::Quaterniond &q_raw, const Eigen::Quaterniond &q_filtered,
                         const Eigen::Quaterniond &q_gt) const;
    void drawInfoText(int x, int y, const std::string &text) const;
    Eigen::Vector3d quaternionToEuler(const Eigen::Quaterniond &q) const;
    void updateCameraPosition();

    TrajectoryData *trajectory_data_;
    int selected_frame_index_;
    float camera_distance_;
    float camera_theta_;
    float camera_phi_;
    Eigen::Vector3d camera_target_;
    bool show_raw_;
    bool show_filtered_;
    bool show_coordinate_frames_;
    bool show_ground_plane_;
    bool show_ground_truth_;
    bool animation_running_;
    int animation_speed_;
    float axis_scale_;
    float robot_scale_;

    int last_mouse_x_;
    int last_mouse_y_;
    bool is_panning_;

    inline static TrajectoryVisualizer *instance_ = nullptr;
};

inline void TrajectoryData::clear()
{
    std::lock_guard<std::mutex> lock(data_mutex);
    raw_trajectory.clear();
    filtered_trajectory.clear();
    ground_truth.clear();
}

inline bool TrajectoryData::saveTumFiles(const std::string &prefix) const
{
    std::lock_guard<std::mutex> lock(data_mutex);
    const bool raw_ok = saveTumFile(prefix + "_raw.tum", raw_trajectory);
    const bool filtered_ok = saveTumFile(prefix + "_filtered.tum", filtered_trajectory);
    const bool gt_ok = saveTumFile(prefix + "_ground_truth.tum", ground_truth);
    return raw_ok && filtered_ok && gt_ok;
}

inline bool TrajectoryData::loadTumFiles(const std::string &raw_path,
                                         const std::string &filtered_path,
                                         const std::string &ground_truth_path)
{
    std::lock_guard<std::mutex> lock(data_mutex);
    bool ok = true;
    ok &= loadTumFile(raw_path, raw_trajectory, false);
    ok &= loadTumFile(filtered_path, filtered_trajectory, true);
    ok &= loadTumFile(ground_truth_path, ground_truth, false);
    return ok;
}

inline bool TrajectoryData::saveTumFile(const std::string &filename, const std::vector<PoseData> &trajectory) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    file << std::fixed << std::setprecision(9);
    for (const auto &pose : trajectory)
    {
        file << pose.timestamp << ' '
             << pose.position.x() << ' '
             << pose.position.y() << ' '
             << pose.position.z() << ' ';

        const Eigen::Quaterniond normalized = pose.orientation.normalized();
        file << normalized.x() << ' '
             << normalized.y() << ' '
             << normalized.z() << ' '
             << normalized.w() << '\n';
    }

    if (trajectory.empty())
    {
        std::cout << "Saved empty TUM trajectory: " << filename << std::endl;
    }
    else
    {
        std::cout << "Saved TUM trajectory: " << filename << std::endl;
    }
    return true;
}

inline bool TrajectoryData::loadTumFile(const std::string &filename,
                                        std::vector<PoseData> &trajectory,
                                        bool is_filtered)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open TUM file: " << filename << std::endl;
        return false;
    }

    trajectory.clear();
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream iss(line);
        double timestamp = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 1.0;

        if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        {
            std::cerr << "Failed to parse line in TUM file: " << line << std::endl;
            return false;
        }

        Eigen::Quaterniond orientation(qw, qx, qy, qz);
        orientation.normalize();

        trajectory.emplace_back(timestamp, Eigen::Vector3d(tx, ty, tz), orientation, is_filtered);
    }

    std::cout << "Loaded TUM trajectory: " << filename << " with " << trajectory.size() << " poses" << std::endl;
    return true;
}

inline TrajectoryVisualizer::TrajectoryVisualizer(TrajectoryData *data)
    : trajectory_data_(data), selected_frame_index_(0), camera_distance_(10.0f),
      camera_theta_(static_cast<float>(M_PI) / 4.0f), camera_phi_(static_cast<float>(M_PI) / 6.0f),
      camera_target_(Eigen::Vector3d::Zero()), show_raw_(true), show_filtered_(true),
      show_coordinate_frames_(true), show_ground_plane_(true), show_ground_truth_(true),
      animation_running_(false), animation_speed_(1), axis_scale_(1.0f), robot_scale_(0.5f),
      last_mouse_x_(0), last_mouse_y_(0), is_panning_(false)
{
}

inline void TrajectoryVisualizer::drawCoordinateFrame(float scale) const
{
    glLineWidth(2.0f);

    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(scale, 0.0f, 0.0f);
    glEnd();

    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, scale, 0.0f);
    glEnd();

    glBegin(GL_LINES);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, scale);
    glEnd();

    glLineWidth(1.0f);
}

inline void TrajectoryVisualizer::drawGrid(int size, float step) const
{
    glColor3f(0.7f, 0.7f, 0.7f);
    glLineWidth(1.0f);

    glBegin(GL_LINES);
    for (int i = -size; i <= size; ++i)
    {
        glVertex3f(static_cast<GLfloat>(i) * step, 0.0f, -static_cast<GLfloat>(size) * step);
        glVertex3f(static_cast<GLfloat>(i) * step, 0.0f, static_cast<GLfloat>(size) * step);

        glVertex3f(-static_cast<GLfloat>(size) * step, 0.0f, static_cast<GLfloat>(i) * step);
        glVertex3f(static_cast<GLfloat>(size) * step, 0.0f, static_cast<GLfloat>(i) * step);
    }
    glEnd();
}

inline void TrajectoryVisualizer::drawRobot(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                                            float size, bool is_filtered) const
{
    glPushMatrix();
    glTranslatef(static_cast<GLfloat>(position.x()), static_cast<GLfloat>(position.y()), static_cast<GLfloat>(position.z()));

    Eigen::AngleAxisd aa(orientation);
    glRotatef(static_cast<GLfloat>(aa.angle() * 180.0 / M_PI), static_cast<GLfloat>(aa.axis().x()),
              static_cast<GLfloat>(aa.axis().y()), static_cast<GLfloat>(aa.axis().z()));

    if (is_filtered)
    {
        glColor3f(0.0f, 0.8f, 0.2f);
    }
    else
    {
        glColor3f(1.0f, 0.4f, 0.4f);
    }

    GLUquadricObj *quadric = gluNewQuadric();
    gluQuadricDrawStyle(quadric, GLU_FILL);
    gluCylinder(quadric, size * 0.8f, size * 0.8f, size * 1.2f, 16, 1);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, size * 1.2f);
    gluDisk(quadric, 0.0, size * 0.8f, 16, 1);
    glPopMatrix();

    gluDisk(quadric, 0.0, size * 0.8f, 16, 1);

    glColor3f(1.0f, 0.0f, 0.0f);
    glPushMatrix();
    glTranslatef(0.0f, 0.0f, size * 0.6f);
    glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
    gluCylinder(quadric, size * 0.1f, size * 0.1f, size * 1.5f, 8, 1);
    glTranslatef(0.0f, 0.0f, size * 1.5f);
    glutSolidCone(size * 0.2f, size * 0.4f, 8, 1);
    glPopMatrix();

    if (show_coordinate_frames_)
    {
        drawCoordinateFrame(size * 2.0f);
    }

    gluDeleteQuadric(quadric);
    glPopMatrix();
}

inline void TrajectoryVisualizer::drawTrajectory(const std::vector<TrajectoryData::PoseData> &trajectory,
                                                 int trajectory_type) const
{
    if (trajectory.empty())
    {
        return;
    }

    float color[3] = {1.0f, 1.0f, 1.0f};
    switch (trajectory_type)
    {
    case 0:
        glColor3f(1.0f, 0.4f, 0.4f);
        color[0] = 1.0f;
        color[1] = 0.4f;
        color[2] = 0.4f;
        glLineWidth(1.5f);
        break;
    case 1:
        glColor3f(0.0f, 0.8f, 0.2f);
        color[0] = 0.0f;
        color[1] = 0.8f;
        color[2] = 0.2f;
        glLineWidth(3.0f);
        break;
    case 2:
        glColor3f(0.2f, 0.4f, 1.0f);
        color[0] = 0.2f;
        color[1] = 0.4f;
        color[2] = 1.0f;
        glLineWidth(2.0f);
        break;
    default:
        break;
    }

    glBegin(GL_LINE_STRIP);
    for (const auto &pose : trajectory)
    {
        glVertex3f(static_cast<GLfloat>(pose.position.x()), static_cast<GLfloat>(pose.position.y()),
                   static_cast<GLfloat>(pose.position.z()));
    }
    glEnd();

    glLineWidth(1.0f);

    const int arrow_interval = 10;
    for (std::size_t i = 0; i < trajectory.size(); i += arrow_interval)
    {
        drawDirectionArrow(trajectory[i].position, trajectory[i].orientation, 0.7f, color, false);
    }

    if (!trajectory.empty() && trajectory_type != 2)
    {
        int frame_index = selected_frame_index_;
        frame_index = std::max(0, std::min(frame_index, static_cast<int>(trajectory.size()) - 1));
        drawRobot(trajectory[frame_index].position, trajectory[frame_index].orientation, robot_scale_,
                  trajectory_type == 1);
    }
    else if (!trajectory.empty() && trajectory_type == 2)
    {
        int frame_index = selected_frame_index_;
        frame_index = std::max(0, std::min(frame_index, static_cast<int>(trajectory.size()) - 1));

        glPushMatrix();
        glTranslatef(static_cast<GLfloat>(trajectory[frame_index].position.x()),
                     static_cast<GLfloat>(trajectory[frame_index].position.y()),
                     static_cast<GLfloat>(trajectory[frame_index].position.z()));
        glColor3f(0.2f, 0.4f, 1.0f);
        glutSolidSphere(robot_scale_ * 0.3f, 12, 12);
        glPopMatrix();
    }
}

inline Eigen::Vector3d TrajectoryVisualizer::quaternionToEuler(const Eigen::Quaterniond &q) const
{
    Eigen::Vector3d euler;
    euler(0) = std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()), 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));

    const double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1.0)
    {
        euler(1) = std::copysign(M_PI / 2.0, sinp);
    }
    else
    {
        euler(1) = std::asin(sinp);
    }

    euler(2) = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));

    euler *= 180.0 / M_PI;
    return euler;
}

inline void TrajectoryVisualizer::drawDirectionArrow(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                                                     float size, const float color[3], bool show_xyz) const
{
    glPushMatrix();
    glTranslatef(static_cast<GLfloat>(position.x()), static_cast<GLfloat>(position.y()), static_cast<GLfloat>(position.z()));

    Eigen::AngleAxisd aa(orientation);
    glRotatef(static_cast<GLfloat>(aa.angle() * 180.0 / M_PI), static_cast<GLfloat>(aa.axis().x()),
              static_cast<GLfloat>(aa.axis().y()), static_cast<GLfloat>(aa.axis().z()));

    glColor3f(color[0], color[1], color[2]);

    GLUquadricObj *quadric = gluNewQuadric();
    gluQuadricDrawStyle(quadric, GLU_FILL);

    glPushMatrix();
    glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
    gluCylinder(quadric, size * 0.05f, size * 0.05f, size, 8, 1);
    glTranslatef(0.0f, 0.0f, size);
    glutSolidCone(size * 0.1f, size * 0.3f, 8, 1);
    glPopMatrix();

    if (show_xyz)
    {
        glPushMatrix();
        glColor3f(color[0] * 0.7f, color[1] * 0.7f, color[2] * 0.7f);
        glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
        gluCylinder(quadric, size * 0.03f, size * 0.03f, size * 0.7f, 8, 1);
        glTranslatef(0.0f, 0.0f, size * 0.7f);
        glutSolidCone(size * 0.08f, size * 0.2f, 8, 1);
        glPopMatrix();

        glPushMatrix();
        glColor3f(color[0] * 0.5f, color[1] * 0.5f, color[2] * 0.5f);
        gluCylinder(quadric, size * 0.03f, size * 0.03f, size * 0.7f, 8, 1);
        glTranslatef(0.0f, 0.0f, size * 0.7f);
        glutSolidCone(size * 0.08f, size * 0.2f, 8, 1);
        glPopMatrix();
    }

    gluDeleteQuadric(quadric);
    glPopMatrix();
}

inline void TrajectoryVisualizer::drawEulerAngles(const Eigen::Quaterniond &q_raw, const Eigen::Quaterniond &q_filtered,
                                                  const Eigen::Quaterniond &q_gt) const
{
    const Eigen::Vector3d euler_raw = quaternionToEuler(q_raw);
    const Eigen::Vector3d euler_filtered = quaternionToEuler(q_filtered);
    const Eigen::Vector3d euler_gt = quaternionToEuler(q_gt);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Raw RPY: [" << euler_raw.x() << ", " << euler_raw.y() << ", " << euler_raw.z() << "] deg  |  ";
    oss << "Filtered RPY: [" << euler_filtered.x() << ", " << euler_filtered.y() << ", " << euler_filtered.z() << "] deg  |  ";
    oss << "GT RPY: [" << euler_gt.x() << ", " << euler_gt.y() << ", " << euler_gt.z() << "] deg";

    drawInfoText(10, glutGet(GLUT_WINDOW_HEIGHT) - 40, oss.str());
}

inline void TrajectoryVisualizer::drawInfoText(int x, int y, const std::string &text) const
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2i(x, y);

    for (const char c : text)
    {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

inline void TrajectoryVisualizer::updateCameraPosition()
{
    const float x = camera_distance_ * std::sin(camera_theta_) * std::cos(camera_phi_);
    const float y = camera_distance_ * std::sin(camera_phi_);
    const float z = camera_distance_ * std::cos(camera_theta_) * std::cos(camera_phi_);

    gluLookAt(camera_target_.x() + x, camera_target_.y() + y, camera_target_.z() + z,
              camera_target_.x(), camera_target_.y(), camera_target_.z(),
              0.0, 1.0, 0.0);
}

inline void TrajectoryVisualizer::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    updateCameraPosition();

    if (show_ground_plane_)
    {
        drawGrid(10, 1.0f);
    }

    drawCoordinateFrame(axis_scale_);

    if (show_ground_truth_ && !trajectory_data_->ground_truth.empty())
    {
        drawTrajectory(trajectory_data_->ground_truth, 2);
    }

    if (show_raw_ && !trajectory_data_->raw_trajectory.empty())
    {
        drawTrajectory(trajectory_data_->raw_trajectory, 0);
    }

    if (show_filtered_ && !trajectory_data_->filtered_trajectory.empty())
    {
        drawTrajectory(trajectory_data_->filtered_trajectory, 1);
    }

    std::string info_text;
    if (!trajectory_data_->raw_trajectory.empty() &&
        selected_frame_index_ < static_cast<int>(trajectory_data_->raw_trajectory.size()))
    {
        std::ostringstream oss;
        oss << "Frame: " << selected_frame_index_;

        const auto &raw = trajectory_data_->raw_trajectory[selected_frame_index_];
        oss << " | Time: " << raw.timestamp;
        oss << " | Raw Pos: (" << raw.position.x() << ", " << raw.position.y() << ", " << raw.position.z() << ")";

        if (!trajectory_data_->ground_truth.empty() &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->ground_truth.size()))
        {
            const auto &gt = trajectory_data_->ground_truth[selected_frame_index_];
            oss << " | GT Pos: (" << gt.position.x() << ", " << gt.position.y() << ", " << gt.position.z() << ")";
        }

        if (!trajectory_data_->filtered_trajectory.empty() &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->filtered_trajectory.size()))
        {
            const auto &filtered = trajectory_data_->filtered_trajectory[selected_frame_index_];
            oss << " | Filtered Pos: (" << filtered.position.x() << ", " << filtered.position.y() << ", " << filtered.position.z() << ")";
        }

        info_text = oss.str();
    }

    if (!info_text.empty())
    {
        drawInfoText(10, glutGet(GLUT_WINDOW_HEIGHT) - 20, info_text);
    }

    if (!trajectory_data_->raw_trajectory.empty() &&
        !trajectory_data_->filtered_trajectory.empty() &&
        !trajectory_data_->ground_truth.empty() &&
        selected_frame_index_ < static_cast<int>(trajectory_data_->raw_trajectory.size()) &&
        selected_frame_index_ < static_cast<int>(trajectory_data_->filtered_trajectory.size()) &&
        selected_frame_index_ < static_cast<int>(trajectory_data_->ground_truth.size()))
    {
        drawEulerAngles(trajectory_data_->raw_trajectory[selected_frame_index_].orientation,
                        trajectory_data_->filtered_trajectory[selected_frame_index_].orientation,
                        trajectory_data_->ground_truth[selected_frame_index_].orientation);
    }

    glutSwapBuffers();
}

inline void TrajectoryVisualizer::handleKeyboard(unsigned char key, int, int)
{
    switch (key)
    {
    case 'r':
        show_raw_ = !show_raw_;
        break;
    case 'f':
        show_filtered_ = !show_filtered_;
        break;
    case 't':
        show_ground_truth_ = !show_ground_truth_;
        break;
    case 'c':
        show_coordinate_frames_ = !show_coordinate_frames_;
        break;
    case 'g':
        show_ground_plane_ = !show_ground_plane_;
        break;
    case '+':
    case '=':
        camera_distance_ = std::max(1.0f, camera_distance_ - 0.5f);
        break;
    case '-':
    case '_':
        camera_distance_ += 0.5f;
        break;
    case 'a':
        animation_running_ = !animation_running_;
        if (animation_running_)
        {
            glutTimerFunc(100, timerCallback, 0);
        }
        break;
    case '1':
        animation_speed_ = std::max(1, animation_speed_ - 1);
        break;
    case '2':
        animation_speed_ += 1;
        break;
    case 's':
        if (trajectory_data_->saveTumFiles("trajectory"))
        {
            std::cout << "Saved trajectories in TUM format with prefix 'trajectory'" << std::endl;
        }
        else
        {
            std::cerr << "Failed to save TUM trajectories" << std::endl;
        }
        break;
    case 'q':
    case 27:
        std::exit(0);
    default:
        break;
    }

    glutPostRedisplay();
}

inline void TrajectoryVisualizer::handleSpecialKeys(int key, int, int)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        selected_frame_index_ = std::max(0, selected_frame_index_ - 1);
        break;
    case GLUT_KEY_RIGHT:
        if (!trajectory_data_->raw_trajectory.empty())
        {
            selected_frame_index_ = std::min(static_cast<int>(trajectory_data_->raw_trajectory.size()) - 1,
                                             selected_frame_index_ + 1);
        }
        break;
    case GLUT_KEY_UP:
        camera_phi_ = std::min(camera_phi_ + 0.05f, static_cast<float>(M_PI) / 2.0f - 0.1f);
        break;
    case GLUT_KEY_DOWN:
        camera_phi_ = std::max(camera_phi_ - 0.05f, -static_cast<float>(M_PI) / 2.0f + 0.1f);
        break;
    default:
        break;
    }

    glutPostRedisplay();
}

inline void TrajectoryVisualizer::handleReshape(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<double>(width) / static_cast<double>(height), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

inline void TrajectoryVisualizer::handleMouse(int button, int state, int x, int y)
{
    last_mouse_x_ = x;
    last_mouse_y_ = y;
    is_panning_ = (button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN);

    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        animation_running_ = false;
    }
}

inline void TrajectoryVisualizer::handleMouseMotion(int x, int y)
{
    const int dx = x - last_mouse_x_;
    const int dy = y - last_mouse_y_;

    if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
    {
        camera_target_.x() -= static_cast<double>(dx) * 0.01;
        camera_target_.z() += static_cast<double>(dy) * 0.01;
    }
    else if (is_panning_)
    {
        camera_target_.x() -= static_cast<double>(dx) * 0.01;
        camera_target_.z() += static_cast<double>(dy) * 0.01;
    }
    else
    {
        camera_theta_ += static_cast<float>(dx) * 0.005f;
        camera_phi_ += static_cast<float>(dy) * 0.005f;
        camera_phi_ = std::clamp(camera_phi_, -static_cast<float>(M_PI) / 2.0f + 0.1f,
                                 static_cast<float>(M_PI) / 2.0f - 0.1f);
    }

    last_mouse_x_ = x;
    last_mouse_y_ = y;

    glutPostRedisplay();
}

inline void TrajectoryVisualizer::advanceFrame()
{
    if (trajectory_data_->raw_trajectory.empty())
    {
        return;
    }

    selected_frame_index_ = (selected_frame_index_ + animation_speed_)
                            % static_cast<int>(trajectory_data_->raw_trajectory.size());
    glutPostRedisplay();
}

inline void TrajectoryVisualizer::setInstance(TrajectoryVisualizer *vis)
{
    instance_ = vis;
}

inline TrajectoryVisualizer *TrajectoryVisualizer::getInstance()
{
    return instance_;
}

inline void TrajectoryVisualizer::timerCallback(int)
{
    TrajectoryVisualizer *instance = getInstance();
    if (instance != nullptr && instance->animation_running_)
    {
        instance->advanceFrame();
        glutTimerFunc(100, timerCallback, 0);
    }
}
