
#include "PCL.h"

using namespace std;

//インプット・アウトプット
pcl::PolygonMesh mesh;
PointCloud cloud(new pcl::PointCloud<pcl::PointXYZ>);
PointCloud cloud2(new pcl::PointCloud<pcl::PointXYZ>);
PointCloud cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

//法線
Normal normals(new pcl::PointCloud<pcl::Normal>);
Normal normals2(new pcl::PointCloud<pcl::Normal>);

//法線情報 point cloud
PointNormal cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);

//キーポイント
PointCloud cloud_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
PointCloud cloud_keypoints2(new pcl::PointCloud<pcl::PointXYZ>);

//reference frame
ReferenceFrame reference_frame(new pcl::PointCloud<pcl::ReferenceFrame>());
ReferenceFrame reference_frame2(new pcl::PointCloud<pcl::ReferenceFrame>());

/*特徴量*/
//spin image
Histogram153 spin_images(new pcl::PointCloud<pcl::Histogram<153> >);
Histogram153 spin_images2(new pcl::PointCloud<pcl::Histogram<153> >);

//shot
SHOT352 shot(new pcl::PointCloud<pcl::SHOT352>());
SHOT352 shot2(new pcl::PointCloud<pcl::SHOT352>());
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
std::vector<pcl::Correspondences> clustered_corrs;

//correspondの情報
pcl::CorrespondencesPtr corrsponds(new pcl::Correspondences());



PCL::PCL(){
	////1つ目のオブジェクトの処理
	iss_SHOT("178_cut.ply",cloud,normals,cloud_keypoints,shot);
	referenceframe_calculation(cloud, cloud_keypoints, normals, reference_frame);

	//２つ目のオブジェクトの処理
	iss_SHOT("177_cut.ply", cloud2, normals2, cloud_keypoints2, shot2);
	referenceframe_calculation(cloud2, cloud_keypoints2, normals2, reference_frame2);

	//一致の計算
	corresponds_calculation_shot(shot,shot2,corrsponds);
	clustering(cloud_keypoints,cloud_keypoints2,reference_frame,reference_frame2,corrsponds,clustered_corrs);
		
	visualize();
}

PCL::~PCL(){

}


void PCL::read(char *name, PointCloud &data){
	pcl::io::loadPLYFile(name, *data);
	cout << "read : " << name << endl;
}

double PCL::computeCloudResolution(PointCloud &data)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

void PCL::filtering(PointCloud input, PointCloud &output){
	//フィルタリング
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(input);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*output);
	cout << "filtering_do"<< endl;
}

void PCL::normal_calculation(PointCloud data, Normal &normal){
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointXYZ>);
	tree_normal->setInputCloud(data);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(data);
	n.setSearchMethod(tree_normal);
	n.setKSearch(20);
	n.compute(*normal);
	cout << "normal_calculation" << endl;
}

void PCL::keypoints_calculation_iss(PointCloud data, PointCloud &keypoint){
	float model_resolution = static_cast<float> (computeCloudResolution(cloud));
	double iss_salient_radius_ = 2 * model_resolution;
	double iss_non_max_radius_ = 4 * model_resolution;
	double iss_gamma_21_(0.975);
	double iss_gamma_32_(0.975);
	double iss_min_neighbors_(5);
	int iss_threads_(4);
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
	
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_keypoint(new pcl::search::KdTree<pcl::PointXYZ>);
	iss_detector.setSearchMethod(tree_keypoint);
	iss_detector.setSalientRadius(iss_salient_radius_);
	iss_detector.setNonMaxRadius(iss_non_max_radius_);
	iss_detector.setThreshold21(iss_gamma_21_);
	iss_detector.setThreshold32(iss_gamma_32_);
	iss_detector.setMinNeighbors(iss_min_neighbors_);
	iss_detector.setNumberOfThreads(iss_threads_);
	iss_detector.setInputCloud(data);
	iss_detector.compute(*keypoint);

	cout << "keypoints_calculation_iss" << endl;

}

void PCL::feature_calculation_spinimage(PointCloud data, Normal normal,Histogram153 &spin_images){
	pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> > spin_image_descriptor(8, 0.5, 16);
	spin_image_descriptor.setInputCloud(data);
	spin_image_descriptor.setInputNormals(normal);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);

	spin_image_descriptor.setSearchMethod(kdtree);
	spin_image_descriptor.setRadiusSearch(0.2);
	spin_image_descriptor.compute(*spin_images);

	cout << "spinimage_calculation" << endl;
}

void PCL::feature_calculation_shot(PointCloud data, PointCloud keypoints, Normal normal, SHOT352 &out){
	pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
	float descr_rad_(0.02f);
	float descr_thread_(4.0f);
	descr_est.setRadiusSearch(descr_rad_);
	descr_est.setNumberOfThreads(descr_thread_);
	descr_est.setInputCloud(keypoints);
	descr_est.setInputNormals(normal);
	descr_est.setSearchSurface(data);
	descr_est.compute(*out);	
}

void PCL::referenceframe_calculation(PointCloud data, PointCloud keypoint, Normal normal, ReferenceFrame &out){
	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
	rf_est.setFindHoles(true);
	float rf_rad_(0.015f);
	rf_est.setRadiusSearch(rf_rad_);

	rf_est.setInputCloud(keypoint);
	rf_est.setInputNormals(normals);
	rf_est.setSearchSurface(data);
	rf_est.compute(*out);
}


void PCL::corresponds_calculation_shot(SHOT352 data, SHOT352 data2, pcl::CorrespondencesPtr &out){
	pcl::KdTreeFLANN<pcl::SHOT352> match_search;
	match_search.setInputCloud(data);

	for (size_t i = 0; i < data2->size(); ++i)
	{
		std::vector<int> neigh_indices(1);
		std::vector<float> neigh_sqr_dists(1);
		if (!pcl_isfinite(data2->at(i).descriptor[0])) //skipping NaNs
		{
			continue;
		}
		int found_neighs = match_search.nearestKSearch(data2->at(i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1 && neigh_sqr_dists[0] < 0.525f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			out->push_back(corr);
		}
	}
	std::cout << "Correspondences found: " << out->size() << std::endl;
}

void PCL::clustering(PointCloud keypoint, PointCloud keypoint2, ReferenceFrame rf, ReferenceFrame rf2, pcl::CorrespondencesPtr corrs, std::vector<pcl::Correspondences> clus_corrs){
	float cg_size_(0.01f);
	float cg_thresh_(5.0f);
	pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
	clusterer.setHoughBinSize(cg_size_);
	clusterer.setHoughThreshold(cg_thresh_);
	clusterer.setUseInterpolation(true);
	clusterer.setUseDistanceWeight(false);

	clusterer.setInputCloud(keypoint);
	clusterer.setInputRf(rf);
	clusterer.setSceneCloud(keypoint2);
	clusterer.setSceneRf(rf2);
	clusterer.setModelSceneCorrespondences(corrs);

	clusterer.recognize(rototranslations, clus_corrs);
	cout << "clustered correspond size :  " << clus_corrs.size() << endl;
}


void PCL::corresponds_display(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer){
	std::cout << "Model instances found: " << rototranslations.size() << std::endl;
	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		cout << "clustered_size" << clustered_corrs[i].size() << endl;
		for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			pcl::PointXYZ& model_point = cloud_keypoints->at(clustered_corrs[i][j].index_query);
			pcl::PointXYZ& scene_point = cloud_keypoints2->at(clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}
}


void PCL::concatenate_field(PointCloud data, Normal normal, PointNormal &cloud_with_normal){
	pcl::concatenateFields(*data, *normal, *cloud_with_normal);
	cout << "concatenate_field" << endl;
}

void PCL::create_mesh(PointCloud data,pcl::PolygonMesh &output ){
	concatenate_field(cloud, normals, cloud_with_normals);


	//k分木作成
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree_mesh(new pcl::search::KdTree<pcl::PointNormal>);
	tree_mesh->setInputCloud(cloud_with_normals);

	//物体の初期処理
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	gp3.setSearchRadius(0.025);//サーチの距離設定

	//パラメータの設定
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	gp3.setMinimumAngle(M_PI / 18); // 10 degrees
	gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
	gp3.setNormalConsistency(false);

	//メッシュ作成
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree_mesh);
	gp3.reconstruct(output);
	cout << "create_mesh" << endl;
}

void PCL::visualize(){
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);

	//色の設定
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud, 100, 100, 100);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler2(cloud2, 100, 100, 100);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler(cloud_keypoints, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler2(cloud_keypoints2, 0, 255, 0);

	//表示するものを選択
	//viewer->addPolygonMesh(mesh, "meshes", 0);
	viewer->addPointCloud(cloud, cloud_color_handler, "cloud");
	viewer->addPointCloud(cloud_keypoints, keypoints_color_handler, "keypoints");
	viewer->addPointCloud(cloud2, cloud_color_handler2, "cloud2");
	viewer->addPointCloud(cloud_keypoints2, keypoints_color_handler2, "keypoints2");

	//corresponds_display(viewer);//特徴点を線で結ぶ
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	while (!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void PCL::iss_SHOT(char *name, PointCloud &PC, Normal &normal, PointCloud &KP, SHOT352 &feature){
	read(name, PC);
	normal_calculation(PC, normal);
	keypoints_calculation_iss(PC, KP);
	feature_calculation_shot(PC, KP, normal, feature);
}