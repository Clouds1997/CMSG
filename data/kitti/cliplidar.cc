#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <map>
#include <fstream>

using namespace std;
using namespace  cv;

std::vector<std::vector<double>> poses;

void getposes(){
    std::string pose_file = "../../data_new/pose/00.txt";

    // 先将pose的数据全部加载进来
    std::string tmp;
    std::fstream file;
    file.open(pose_file);
    while(getline(file, tmp)){
        vector<double> pose;
        stringstream ss(tmp);
        string word;
        while(ss >> word){
            pose.push_back(stod(word));
        }
        poses.push_back(pose);
    }
    std::cout << "get the pose "<<poses.size() <<std::endl;
}



void process(string name){
    // 加载lidar点云
    std::string str = "../../kitti/00/semantic_lidar/" +name + ".pcd";

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);
    if (pcl::io::loadPCDFile<pcl::PointXYZL> (str, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudB.pcd \n");
        // return (-1);
        return;
    }
    std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from cloudB.pcd with the following fields: "
            << std::endl;

    // std::vector<Eigen::Vector3d> lidarPoints;
    pcl::PointCloud<pcl::PointXYZL> sumCloud;
    
    for(int i = 0; i < cloud->size(); ++ i){
        double X  = cloud->points[i].x;
        double Y   = cloud->points[i].y;
        double Z   = cloud->points[i].z;
        double dis_sqr = X * X  + Y * Y + Z * Z;
        // lidarPoints.push_back(point);
        if (X >= 5 && X < 30 && Y >= -14 && Y < 14)
            { 
                pcl::PointXYZL pt_xyzl;
                pt_xyzl.x = cloud->points[i].x;
                pt_xyzl.y = cloud->points[i].y;
                pt_xyzl.z = cloud->points[i].z;
                pt_xyzl.label = cloud->points[i].label;

                sumCloud.points.push_back(pt_xyzl);
            }
    }
    sumCloud.height = 1;
    sumCloud.width = sumCloud.points.size();
    pcl::io::savePCDFileASCII("../../kitti/00/semantic_lidar_clip_530/"+ name + ".pcd", sumCloud);
    std::cout << "success save the " + name + ".pcd"  << std::endl;

}

int main(){
    getposes();
    int nums = poses.size();

    // get_imgtimestamp();
    // get_lidartimestamp();

    // cout << "get " << imgtimestamp.size() << "of img and " <<lidartimestamp.size() <<" of lidar" <<endl;

    for(int i = 0; i < nums; i ++){
        stringstream ss;
        ss << setw(10) << setfill('0') << i;
        string str;
        ss >> str;
        // std::cout << std::stoi(str) << std::endl;
        process(str);
    }
    return 0;
}