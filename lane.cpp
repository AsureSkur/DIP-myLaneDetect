#include<cstdio>
#include<algorithm>
#include<cstdlib>
#include<vector>
#include<list>
#include<cstring>
#include<opencv.hpp>
#include<fstream>
#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <io.h>
//Hough
typedef struct line
{
	double r;//极径
	double theta;//极角，角度
}hline;
int line_cal(struct line l, int y);
void hough_line(cv::Mat& img,int color, int threshold, std::vector<line>& lines)
{
	int** board;
	int size = img.cols * img.cols + img.rows * img.rows;
	int max = (int)sqrt(size);
	size = 2 * sqrt(size) + 100;
	board = (int**)malloc(size * sizeof(int*));
	for (int i = 0; i < size; i++)
	{
		board[i] = (int*)malloc(181 * sizeof(int));
		memset(board[i], 0, 181 * sizeof(int));
	}
	cv::Vec3b pix;
	double PI = 3.1415926535;
	//vote
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			int pix = img.at<uchar>(i, j);
			if (pix)
			{
				for (int theta = 0; theta < 181; theta++)
				{
					int r = j * cos((double)theta * PI / 180.0) + i * sin((double)theta * PI / 180.0);
					if (r < 0)
					{
						continue;
					}
					board[r][theta] += 1;
					if (pix > color)
					{
						board[r][theta] += 1;
					}
				}
			}
		}
	}
	//visit all board, select line
	struct line tmp;
	for (int i = 0; i < size; i++)//p
	{
		for (int j = 0; j < 181; j++)//angle
		{
			if (board[i][j] >= threshold)
			{
				if (i < 10 || i > max - 10)
				{
					continue;
				}
				if (j > 170 || j < 10)
				{
					if (i < max / 10)
					{
						continue;
					}
					if (i > img.rows * 11 / 10 || i < img.rows * 9 / 10)
					{
						continue;
					}
					if (i > img.cols * 11 / 10 || i < img.cols * 9 / 10)
					{
						continue;
					}
				}
				tmp.r = i;
				tmp.theta = j;
				int cnt = 0;
				for (int x = 160; x <= 710; x ++)
				{
					int y = line_cal(tmp, x);
					if (y < 0)
					{
						continue;
					}
					else if (y >= img.cols)
					{
						continue;
					}
					else
					{
						if (img.at<uchar>(x, y))
						{
							cnt++;
						}
					}
				}
				if (cnt < threshold/2)
				{
					continue;
				}
				lines.push_back(tmp);
			}
		}
	}
	printf("Hough line num:%d\n", lines.size());
	//free
	for (int i = 0; i < size; i++)
	{
		free(board[i]);
	}
	free(board);
	//debug
	return;
}
//canny
void generateGaussMask(cv::Mat& Mask, cv::Size wsize, double sigma)
{
	Mask.create(wsize, CV_64F);
	double sum = 0.0;
	int x0, y0;//axis of the center point
	x0 = (wsize.height - 1) / 2;
	y0 = (wsize.width - 1) / 2;
	double x, y;
	for (int i = 0; i < wsize.height; i++)
	{
		y = ((double)i - y0) * ((double)i - y0);
		for (int j = 0; j < wsize.width; j++)
		{
			x = ((double)j - x0) * ((double)j - x0);
			double tmp = exp(-(x + y) / (2 * sigma * sigma));
			Mask.at<double>(i, j) = tmp;
			sum += tmp;
		}
	}
	Mask = Mask / sum;
}
void Filter(cv::Mat src, cv::Mat& dst, cv::Mat filter)
{
	int st_h = (filter.rows - 1) / 2;
	int st_w = (filter.cols - 1) / 2;
	int ed_h = src.rows - st_h;
	int ed_w = src.cols - st_w;
	dst = cv::Mat::zeros(src.size(), src.type());//do not expand the picture, leave the border as it used to be

	for (int y = st_h; y < ed_h; y++)
	{
		for (int x = st_w; x < ed_w; x++)
		{
			double sum[3] = { 0 };
			for (int i = 0; i < filter.rows; i++)//y axis of mask matrix
			{
				for (int j = 0; j < filter.cols; j++)//x axis of mask matrix
				{
					int y_t, x_t;
					y_t = y - st_h + i;
					x_t = x - st_w + j;
					if (src.channels() == 1) //gray pic
					{
						sum[0] = sum[0] + src.at<uchar>(y_t, x_t) * filter.at<double>(i, j);
					}
					else if (src.channels() == 3) //rgb pic
					{
						cv::Vec3b rgb = src.at<cv::Vec3b>(y_t, x_t);
						sum[0] += rgb[0] * filter.at<double>(i, j);
						sum[1] += rgb[1] * filter.at<double>(i, j);
						sum[2] += rgb[2] * filter.at<double>(i, j);
					}

				}
			}
			for (int k = 0; k < src.channels(); k++)
			{
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(y, x) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(y, x) = rgb;
			}
		}
	}
}

void Sobel(const cv::Mat src, cv::Mat& dst, double* direction)
{
	cv::Mat sobel_x = cv::Mat::zeros(src.size(), CV_32SC1);
	cv::Mat sobel_y = cv::Mat::zeros(src.size(), CV_32SC1);
	uchar* P = src.data;
	uchar* PX = sobel_x.data;
	uchar* PY = sobel_y.data;

	int step = src.step;
	int stepXY = sobel_x.step;
	int k = 0;
	int m = 0;
	int n = 0;
	for (int i = 1; i < (src.rows - 1); i++)
	{
		for (int j = 1; j < (src.cols - 1); j++)
		{
			double gradY = P[(i - 1) * step + j + 1] + P[i * step + j + 1] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[i * step + j - 1] * 2 - P[(i + 1) * step + j - 1];
			PY[i * stepXY + j * (stepXY / step)] = abs(gradY);
			double gradX = P[(i + 1) * step + j - 1] + P[(i + 1) * step + j] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[(i - 1) * step + j] * 2 - P[(i - 1) * step + j + 1];
			PX[i * stepXY + j * (stepXY / step)] = abs(gradX);
			if (gradX == 0)
			{
				gradX = 0.00000000000000001;
			}
			direction[k] = atan(gradY / gradX) * 57.3;//弧度转换为度
			direction[k] += 90;
			k++;
		}
	}
	//trans cv_32u to cv_8u
	convertScaleAbs(sobel_x, sobel_x);
	convertScaleAbs(sobel_y, sobel_y);
	dst = cv::Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<float>(i, j) = sqrt(sobel_x.at<uchar>(i, j) * sobel_x.at<uchar>(i, j) + sobel_y.at<uchar>(i, j) * sobel_y.at<uchar>(i, j));
		}
	}
	//trans cv_32u to cv_8u
	convertScaleAbs(dst, dst);
}
void LocalMaxValue(const cv::Mat src, cv::Mat& dst, double* direction)
{
	dst = src.clone();
	int k = 0;
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			int value00 = src.at<uchar>((i - 1), j - 1);
			int value01 = src.at<uchar>((i - 1), j);
			int value02 = src.at<uchar>((i - 1), j + 1);
			int value10 = src.at<uchar>((i), j - 1);
			int value11 = src.at<uchar>((i), j);
			int value12 = src.at<uchar>((i), j + 1);
			int value20 = src.at<uchar>((i + 1), j - 1);
			int value21 = src.at<uchar>((i + 1), j);
			int value22 = src.at<uchar>((i + 1), j + 1);

			if (direction[k] > 0 && direction[k] <= 45)
			{
				if (value11 <= (value12 + (value02 - value12) * tan(direction[i * dst.rows + j])) || (value11 <= (value10 + (value20 - value10) * tan(direction[i * dst.rows + j]))))
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
			if (direction[k] > 45 && direction[k] <= 90)

			{
				if (value11 <= (value01 + (value02 - value01) / tan(direction[i * (dst.cols - 1) + j])) || value11 <= (value21 + (value20 - value21) / tan(direction[i * (dst.cols - 1) + j])))
				{
					dst.at<uchar>(i, j) = 0;

				}
			}
			if (direction[k] > 90 && direction[k] <= 135)
			{
				if (value11 <= (value01 + (value00 - value01) / tan(180 - direction[i * (dst.cols - 1) + j])) || value11 <= (value21 + (value22 - value21) / tan(180 - direction[i * (dst.cols - 1) + j])))
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
			if (direction[k] > 135 && direction[k] <= 180)
			{
				if (value11 <= (value10 + (value00 - value10) * tan(180 - direction[i * (dst.cols-1) + j])) || value11 <= (value12 + (value22 - value11) * tan(180 - direction[i * (dst.cols - 1) + j])))
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
			k++;
		}
	}
	return;
}
void DoubleThreshold(cv::Mat& src, double lowThreshold, double highThreshold)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) > highThreshold)
			{
				src.at<uchar>(i, j) = 255;
			}
			if (src.at<uchar>(i, j) < lowThreshold)
			{
				src.at<uchar>(i, j) = 0;
			}
		}
	}
	return;
}
void DoubleThresholdLink(cv::Mat& src, double lowThreshold, double highThreshold)
{
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			if (src.at<uchar>(i, j) > lowThreshold && src.at<uchar>(i, j) < 255)
			{
				if (src.at<uchar>(i - 1, j - 1) == 255 || src.at<uchar>(i - 1, j) == 255 || src.at<uchar>(i - 1, j + 1) == 255 ||
					src.at<uchar>(i, j - 1) == 255 || src.at<uchar>(i, j) == 255 || src.at<uchar>(i, j + 1) == 255 ||
					src.at<uchar>(i + 1, j - 1) == 255 || src.at<uchar>(i + 1, j) == 255 || src.at<uchar>(i + 1, j + 1) == 255)
				{
					src.at<uchar>(i, j) = 255;
					//DoubleThresholdLink(src, lowThreshold, highThreshold);
				}
				else
				{
					src.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return;
}

void canny(cv::Mat& img_src,cv::Mat& dst)
{
	cv::Mat mask_gauss;
	generateGaussMask(mask_gauss, cv::Size(5, 5), 1);
	cv::Mat img_gauss = cv::Mat::zeros(img_src.size(), CV_8UC1);
	Filter(img_src, img_gauss, mask_gauss);
	//cv::imshow("Gaussian", img_gauss);
	//cv::waitKey(0);
	cv::Mat img_sobel;
	double* direction = (double*)malloc((img_src.rows - 1)*(img_src.cols - 1)*sizeof(double));
	for (int i = 0; i < (img_src.rows - 1) * (img_src.cols - 1); i++)
	{
		direction[i] = 0;
	}
	Sobel(img_gauss, img_sobel, direction);
	//cv::imshow("Sobel", img_sobel);
	//cv::waitKey(0);
	dst = cv::Mat::zeros(img_sobel.size(), CV_8UC1);
	LocalMaxValue(img_sobel, dst, direction);
	//cv::imshow("LocalMax", dst);
	//cv::waitKey(0);
	DoubleThreshold(dst, 90, 160);
	//DoubleThresholdLink(dst, 90, 160);

	//cv::imshow("Canny", dst);
	//cv::waitKey(0);
	free(direction);
	return;
}

//binary
void binary(cv::Mat& img, double threshold = 120)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<double>(i, j) = img.at<double>(i, j) > threshold ? 255 : 0;
		}
	}
	return;
}

//k-means
bool cmp_r(struct line a,struct line b)
{
	if (a.theta > b.theta)
	{
		return 1;
	}
	else if (a.theta == b.theta)
	{
		return a.r > b.r;
	}
	else return 0;
}
double dist_calc(struct line a, struct line b, int mapsize)
{
	if (a.r == -1 || b.r == -1)
	{
		return INFINITY;
	}
	double dst;
	a.r /= mapsize;
	b.r /= mapsize;
	a.theta /= 60;
	b.theta /= 60;
	dst = sqrt((a.r - b.r) * (a.r - b.r) + (a.theta - b.theta) * (a.theta - b.theta));
	return dst;
}
void k_means(std::vector<line>src, std::vector<line>&dst,int mapsize, int type = 6)
{
	if (src.size() == 0)
	{
		dst.clear();
		return;
	}
	sort(src.begin(), src.end(), cmp_r);
	int step = src.size() / type;
	struct line cluster[6];
	//init
	for (int i = 0; i < type; i++)
	{
		cluster[i].r = src[step * i].r;
		cluster[i].theta = src[step * i].theta;
	}
	double* dist = (double*)malloc(sizeof(double) * src.size());
	for (int i = 0; i < src.size(); i++)
	{
		dist[i] = INFINITY;
	}
	int* flag = (int*)malloc(sizeof(int) * src.size());
	for (int i = 0; i < src.size(); i++)
	{
		flag[i] = -1;
	}
	double error = 0.0;
	double limit = 1;
	while (1)
	{
		error = 0.0;
		//divide
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < src.size(); j++)
			{
				double dist_tmp = dist_calc(cluster[i], src[j], mapsize);
				if (dist_tmp < dist[j])
				{
					dist[j] = dist_tmp;
					flag[j] = i;
				}
			}
		}
		//calculate center
		double c_r[6];
		double c_theta[6];
		int c_num[6];
		memset(c_r, 0, sizeof(c_r));
		memset(c_theta, 0, sizeof(c_theta));
		memset(c_num, 0, sizeof(c_num));
		for (int i = 0; i < src.size(); i++)
		{
			if (flag[i] == -1)
			{
				printf("Error!\n");
				continue;
			}
			c_r[flag[i]] += src[i].r;
			c_theta[flag[i]] += src[i].theta;
			c_num[flag[i]] += 1;
		}
		for (int i = 0; i < type; i++)
		{
			if (c_num[i] <= 0)
			{
				cluster[i].r = -1;
				continue;
			}
			c_r[i] /= c_num[i];
			c_theta[i] /= c_num[i];
			error += (cluster[i].r - c_r[i]) * (cluster[i].r - c_r[i]) + (cluster[i].theta - c_theta[i]) * (cluster[i].theta -c_theta[i]);
			cluster[i].r = c_r[i];
			cluster[i].theta = c_theta[i];
		
		}
		printf("K-mean error:%lf\n", error);
		if (error <= limit)
		{
			break;
		}
	}
	free(dist);
	free(flag);
	dst.clear();
	for (int i = 0; i < type; i++)
	{
		if (cluster[i].r > 0)
		{
			dst.push_back(cluster[i]);
		}
	}
	return;
}

//write predict into json
//line: calculate x
int line_cal(struct line l, int y)
{
	double PI = 3.1415926535;
	double a = cos((double)l.theta * PI / 180.0);
	double b = sin((double)l.theta * PI / 180.0);
	double x0 = a * l.r;
	double y0 = b * l.r;
	double dy = y - y0;
	double dx = -tan((double)l.theta * PI / 180.0) * dy;
	return (int)(x0 + dx);
}
void json_write(const char* filepath, std::vector<line>lines, FILE* fp,cv::Mat canny)
{
	if (lines.size() == 0)
	{
		return;
	}
	fprintf(fp, "{\"lanes\": [");
	int flag = 0;
	for (int k = 0; k < lines.size(); k++)
	{
		flag = 0;
		if (k != 0)
		{
			fprintf(fp, ", ");
		}
		fprintf(fp, "[");
		for (int i = 160; i <= 710; i += 10)
		{
			int j = line_cal(lines[k], i);
			if (j < 0)
			{
				j = -2;
			}
			else if (j >= canny.cols)
			{
				j = -2;
			}
			else
			{
				if (canny.at<uchar>(i, j))
				{
					flag = 1;
				}
			}
			if (!flag)
			{
				j = -2;
			}
			if (i != 160)
			{
				fprintf(fp, ", ");
			}
			fprintf(fp, "%d", j);
		}
		fprintf(fp, "]");
	}
	fprintf(fp, "], \"h_samples\": [");
	for (int i = 160; i <= 710; i += 10)
	{
		if (i != 160)
		{
			fprintf(fp, ", ");
		}
		fprintf(fp, "%d", i);
	}
	fprintf(fp, "], \"raw_file\": \"%s\"}\n", filepath);

}

//lane detection
void lane(const char* filepath)
{
	double PI = 3.1415926535;
	std::string img_path;
	img_path = filepath;
	cv::Mat srcimg = cv::imread(img_path, 0);
	cv::Mat cannyimg;
	canny(srcimg, cannyimg);
	std::vector<line>lines;
	hough_line(cannyimg, 90, 50, lines);
	srcimg = cv::imread(img_path);
	for (int i = 0; i < lines.size(); i++)
	{
		double a = cos((double)lines[i].theta * PI / 180.0);
		double b = sin((double)lines[i].theta * PI / 180.0);
		double x0 = a * lines[i].r;
		double y0 = b * lines[i].r;
		int x1 = x0 + 10000 * (-b);
		int y1 = y0 + 10000 * (a);
		int x2 = x0 - 10000 * (-b);
		int y2 = y0 - 10000 * (a);
		cv::line(srcimg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0));
	}
	//cv::imshow("Hough", srcimg);
	//cv::waitKey(0);
	//k-means as k=6
	double size = sqrt(srcimg.cols * srcimg.cols + srcimg.rows * srcimg.rows);
	k_means(lines, lines, (int)size);

	//srcimg = cv::imread(img_path);
	for (int i = 0; i < lines.size(); i++)
	{
		double a = cos((double)lines[i].theta * PI / 180.0);
		double b = sin((double)lines[i].theta * PI / 180.0);
		double x0 = a * lines[i].r;
		double y0 = b * lines[i].r;
		int x1 = x0 + 10000 * (-b);
		int y1 = y0 + 10000 * (a);
		int x2 = x0 - 10000 * (-b);
		int y2 = y0 - 10000 * (a);
		cv::line(srcimg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255),5);
	}
	cv::imshow("Final", srcimg);
	cv::waitKey(0);
	//write into json
	FILE* fpWrite = fopen("pred.json", "ab");
	json_write(filepath, lines, fpWrite, cannyimg);
	fclose(fpWrite);
	return;
}

//get filename
void GetJustCurrentDir(std::string path, std::vector<std::string>& files)
{
	//文件信息 
	struct _finddata_t fileinfo;
	std::string strP;
	long long  hFile = 0;
	if ((hFile = _findfirst(strP.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(fileinfo.name);
				}

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	else
	{
		long hFile = 0;
		if ((hFile = _findfirst(strP.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				if ((fileinfo.attrib & _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					{
						files.push_back(fileinfo.name);
					}

				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
	}
}
int main()
{
	FILE* fpWrite = fopen("pred.json", "w+");
	fclose(fpWrite);
	std::vector<std::string> files;
	GetJustCurrentDir("clips/0531", files);
	char front[260] = "clips/";
	char behind[260] = "/20.jpg";
	for (int i = 0; i < files.size(); i++)
	{
		//debug
		printf("NO:%d\n", i);

		char filepath[260] = "";
		strcpy(filepath, front);
		strcat(filepath, "0531/");
		strcat(filepath, files[i].c_str());
		strcat(filepath, behind);
		lane(filepath);
	}
	files.clear();
	GetJustCurrentDir("clips/0601", files);
	for (int i = 0; i < files.size(); i++)
	{
		//debug
		printf("NO:%d\n", i);

		char filepath[260] = "";
		strcpy(filepath, front);
		strcat(filepath, "0601/");
		strcat(filepath, files[i].c_str());
		strcat(filepath, behind);
		lane(filepath);
	}
	
	//lane("pic.jpg");
	return 0;
}