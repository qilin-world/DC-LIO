#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;
    int cornerTargetNum = 0;
    int surfaceTargetNum = 0;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        
        initializationValue();
    }

private:
    inline float clampRatioValue(float ratio) const
    {
        if (ratio < 0.0f) return 0.0f;
        if (ratio > 1.0f) return 1.0f;
        return ratio;
    }

    void updateFeatureExtractionTargets()
    {
        int cloudSize = extractedCloud->points.size();
        if (cloudSize <= 0){
            cornerTargetNum = 0;
            surfaceTargetNum = 0;
            return;
        }

        if (!enableAdaptiveFeatureSelection){
            cornerTargetNum = cloudSize;
            surfaceTargetNum = cloudSize;
            return;
        }

        float cornerRatio = clampRatioValue(cornerFeatureRatio);
        float surfaceRatio = clampRatioValue(surfaceFeatureRatio);

        cornerTargetNum = std::max(static_cast<int>(cornerRatio * cloudSize), edgeFeatureMinValidNum);
        cornerTargetNum = std::min(cornerTargetNum, cloudSize);

        int remainingAfterCorner = std::max(cloudSize - cornerTargetNum, 0);
        surfaceTargetNum = std::max(static_cast<int>(surfaceRatio * cloudSize), surfFeatureMinValidNum);
        surfaceTargetNum = std::min(surfaceTargetNum, remainingAfterCorner);
    }

public:

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        printf("cloud size: %d\n", extractedCloud->size());

        updateFeatureExtractionTargets();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        const float rangeWeight = clampRatioValue(curvatureRangeWeight);
        const float xyzWeight = 1.0f - rangeWeight;

        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            float rangeCurvature = diffRange * diffRange;

            float diffX = 0.0f;
            float diffY = 0.0f;
            float diffZ = 0.0f;
            for (int offset = -5; offset <= 5; ++offset){
                if (offset == 0) continue;
                diffX += extractedCloud->points[i + offset].x;
                diffY += extractedCloud->points[i + offset].y;
                diffZ += extractedCloud->points[i + offset].z;
            }
            diffX -= extractedCloud->points[i].x * 10;
            diffY -= extractedCloud->points[i].y * 10;
            diffZ -= extractedCloud->points[i].z * 10;
            float xyzCurvature = diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudCurvature[i] = rangeWeight * rangeCurvature + xyzWeight * xyzCurvature;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        int totalSurfaceSelections = 0;

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep+1, by_value());

                int segmentSize = ep - sp + 1;
                const int legacyCornerBudget = 20;
                int globalCornerRemaining = enableAdaptiveFeatureSelection ? std::max(cornerTargetNum - static_cast<int>(cornerCloud->size()), 0) : std::numeric_limits<int>::max();
                int globalSurfaceRemaining = enableAdaptiveFeatureSelection ? std::max(surfaceTargetNum - totalSurfaceSelections, 0) : std::numeric_limits<int>::max();
                int segmentCornerBudget = enableAdaptiveFeatureSelection ?
                    std::min(std::max(static_cast<int>(segmentSize * clampRatioValue(cornerFeatureRatio)), 1), std::max(globalCornerRemaining, 0)) :
                    legacyCornerBudget;
                int segmentSurfaceBudget = enableAdaptiveFeatureSelection ?
                    std::min(std::max(static_cast<int>(segmentSize * clampRatioValue(surfaceFeatureRatio)), 1), std::max(globalSurfaceRemaining, 0)) :
                    std::numeric_limits<int>::max();
                int segmentCornerPicked = 0;
                int segmentSurfacePicked = 0;

                if (!enableAdaptiveFeatureSelection || globalCornerRemaining > 0){
                    int largestPickedNum = 0;
                    for (int k = ep; k >= sp; k--)
                    {
                        int ind = cloudSmoothness[k].ind;
                        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                        {
                            largestPickedNum++;
                            if (largestPickedNum <= segmentCornerBudget){
                                cloudLabel[ind] = 1;
                                cornerCloud->push_back(extractedCloud->points[ind]);
                                segmentCornerPicked++;
                            } else {
                                break;
                            }

                            cloudNeighborPicked[ind] = 1;
                            for (int l = 1; l <= 5; l++)
                            {
                                int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                                if (columnDiff > 10)
                                    break;
                                cloudNeighborPicked[ind + l] = 1;
                            }
                            for (int l = -1; l >= -5; l--)
                            {
                                int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                                if (columnDiff > 10)
                                    break;
                                cloudNeighborPicked[ind + l] = 1;
                            }
                        }

                        if (enableAdaptiveFeatureSelection && (segmentCornerPicked >= segmentCornerBudget || static_cast<int>(cornerCloud->size()) >= cornerTargetNum))
                            break;
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (globalSurfaceRemaining <= 0 && enableAdaptiveFeatureSelection)
                        break;

                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;
                        segmentSurfacePicked++;
                        totalSurfaceSelections++;
                        globalSurfaceRemaining = enableAdaptiveFeatureSelection ? std::max(surfaceTargetNum - totalSurfaceSelections, 0) : std::numeric_limits<int>::max();

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }

                        if (enableAdaptiveFeatureSelection && (segmentSurfacePicked >= segmentSurfaceBudget))
                            break;
                    }
                }

                if (enableAdaptiveFeatureSelection && surfaceTargetNum > 0 && totalSurfaceSelections >= surfaceTargetNum){
                    // still push already selected surfaces for this segment before exiting
                    ;
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (!enableAdaptiveFeatureSelection){
                        if (cloudLabel[ind] <= 0)
                            surfaceCloudScan->push_back(extractedCloud->points[ind]);
                    }else{
                        if (cloudLabel[ind] == -1)
                            surfaceCloudScan->push_back(extractedCloud->points[ind]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}