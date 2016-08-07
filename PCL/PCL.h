#ifndef PCL_H_
#define PCL_H_
#pragma warning(disable:4996)


#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/file_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/spin_image.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include "PCLAdapter.h"
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud;
typedef pcl::PointCloud<pcl::Normal>::Ptr Normal;
typedef pcl::PointCloud<pcl::ReferenceFrame>::Ptr ReferenceFrame;
typedef pcl::PointCloud<pcl::Histogram<153> >::Ptr Histogram153;
typedef pcl::PointCloud<pcl::PointNormal>::Ptr PointNormal;
typedef pcl::PointCloud<pcl::SHOT352>::Ptr SHOT352;

class PCL{
private:
	//�ǂݍ���
	void read(char *name, PointCloud &data);

	//�t�B���^�����O
	void filtering(PointCloud input, PointCloud &output);

	//resolution�̌v�Z
	double computeCloudResolution(PointCloud &data);

	//�@���v�Z
	void normal_calculation(PointCloud data, Normal &normal);

	//�@����� point cloud
	void concatenate_field(PointCloud data, Normal normal, PointNormal &cloud_with_normal);

	//�L�[�|�C���g�̌v�Z
	void keypoints_calculation_iss(PointCloud data, PointCloud &keypoint);

	//referenceframe
	void referenceframe_calculation(PointCloud data, PointCloud keypoint, Normal normal, ReferenceFrame &out);

	/*feature�v�Z*/
	//spinimages
	void feature_calculation_spinimage(PointCloud data, Normal normal,Histogram153 &spin_images);
	//shot
	void feature_calculation_shot(PointCloud data, PointCloud keypoint, Normal normal, SHOT352 &out);

	//�N���X�^�����O
	void clustering(PointCloud keypoint, PointCloud keypoint2, ReferenceFrame rf, ReferenceFrame rf2, pcl::CorrespondencesPtr corrs, std::vector<pcl::Correspondences> clus_corrs);

	//�Ή�����
	void corresponds_calculation_shot(SHOT352 data, SHOT352 data2, pcl::CorrespondencesPtr &out);
	void corresponds_display(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer);

	//���b�V���쐬
	void create_mesh(PointCloud data, pcl::PolygonMesh &output);

	//����
	void visualize();

	//iss�ŃL�[�|�C���g�ASHOT�œ�����
	void iss_SHOT(char *name, PointCloud &PC, Normal &normal, PointCloud &KP,SHOT352 &feature);

public:
	PCL();
	~PCL();
};


#endif