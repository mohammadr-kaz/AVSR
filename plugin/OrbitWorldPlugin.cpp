#include <ignition/math/Pose3.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"

#include "ros/ros.h"
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Quaternion.h>
#include <iostream>


namespace gazebo
{
    class EarthOrbit: public WorldPlugin
    {
        private:
            ros::NodeHandle* nodeHandle;
            ros::Subscriber earthStateSubscriber;
            ros::Subscriber chaserStateSubscriber;
            ros::Subscriber targetStateSubscriber;
            ros::Subscriber earthResetOrientation;

            // Pointer to model
            physics::ModelPtr earthModel;
            physics::ModelPtr chaserModel;
            physics::ModelPtr targetModel;

            physics::LinkPtr chaserBaseLink;

            geometry_msgs::Vector3 earthVelocityRelativeToNED;

            ignition::math::Pose3d chaserPoseRelativeToTargetNED;

            ignition::math::Quaterniond earthOrientationRelativeToNED;

            ignition::math::Vector3d earthPositionRelativeToNED;

            //Pointer to update event connection

            event::ConnectionPtr updateConnection;
            

        public:
            void Load(physics::WorldPtr _parent, sdf::ElementPtr )
            {
                this->earthModel = _parent->ModelByName("earth");
                this->targetModel = _parent->ModelByName("target");
                this->chaserModel = _parent->ModelByName("chaser");
                this->chaserBaseLink = this->chaserModel->GetLink("link");

                this->chaserPoseRelativeToTargetNED = this->chaserModel->WorldPose();
                this->earthPositionRelativeToNED = this->earthModel->WorldPose().Pos();

                this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&EarthOrbit::onUpdate,
                                                                                          this));

                int argc = 0;
                char **argv;

                ros::init(argc, argv, "orbitWorldNode");
                this->nodeHandle = new ros::NodeHandle("orbitWorldNode");

                this->earthStateSubscriber = this->nodeHandle->subscribe("/earthState", 1, &EarthOrbit::earthCallback, this);
                this->chaserStateSubscriber = this->nodeHandle->subscribe("/chaserState", 1, &EarthOrbit::chaserCallback, this);
                this->targetStateSubscriber = this->nodeHandle->subscribe("/targetState", 1, &EarthOrbit::targetCallback, this);
                this->earthResetOrientation = this->nodeHandle->subscribe("earthReset", 1, &EarthOrbit::earthResetCallback, this);


                std::cout<<"plugin loaded.\n";

            };

            void onUpdate()
            {
                float wx = this->earthVelocityRelativeToNED.x;
                float wy = this->earthVelocityRelativeToNED.y;
                float wz = this->earthVelocityRelativeToNED.z;

                this->earthModel->SetAngularVel(ignition::math::Vector3d(wx, wy, wz));

                this->chaserBaseLink->SetWorldPose(this->chaserPoseRelativeToTargetNED);
            };

            void earthCallback(const geometry_msgs::Vector3::ConstPtr& msg)
            {
                this->earthVelocityRelativeToNED = *msg;
            };

            void earthResetCallback(const geometry_msgs::Quaternion::ConstPtr &msg)
            {
                float qx = msg->x;
                float qy = msg->y;
                float qz = msg->z;
                float qw = msg->w;

                ignition::math::Quaterniond rot(qw, qx, qy, qz);

                this->earthModel->SetWorldPose(ignition::math::Pose3d(this->earthPositionRelativeToNED, rot));
            };

            void chaserCallback(const geometry_msgs::Pose::ConstPtr &msg)
            {
                float x = msg->position.x;
                float y = msg->position.y;
                float z = msg->position.z;

                ignition::math::Vector3d position(x, y, z);

                float qx = msg->orientation.x;
                float qy = msg->orientation.y;
                float qz = msg->orientation.z;
                float qw = msg->orientation.w;

                ignition::math::Quaterniond rot(qw, qx, qy, qz);

                this->chaserPoseRelativeToTargetNED = ignition::math::Pose3d(position, rot);
            };

            void targetCallback(const geometry_msgs::Quaternion::ConstPtr &msg)
            {
                float qx = msg->x;
                float qy = msg->y;
                float qz = msg->z;
                float qw = msg->w;

                ignition::math::Quaterniond rot(qw, qx, qy, qz);

                this->targetModel->SetWorldPose(ignition::math::Pose3d(ignition::math::Vector3d(0, 0, 0), rot));
            };

    };

    // Register this plugin with the simulator.
    GZ_REGISTER_WORLD_PLUGIN(EarthOrbit)
} 