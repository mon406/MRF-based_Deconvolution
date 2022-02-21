#ifndef __INCLUDED_H_Deconvolution__
#define __INCLUDED_H_Deconvolution__

#include "main.h"
#include "Fourier_Transform_Mat.h"
#include "Calculate_Complex_Number.h"
#include "Image_Evaluation.h"			// �摜�̕]��(MSE, PSNR, SSIM)

/* �֐��錾 */
void multi_Matrix_Vector_3D(int fft_x, int fft_y, FTMat3D& InputImg, FTMat& dft_InputTransVector, FTMat3D& OutputImg);		// 2�����x�N�g����1�����x�N�g���̏�Z
double multi_vector(Mat& srcImg, Mat& srcImg2);							// 1�����x�N�g���̏�Z(����)


/*--- MRF_Deconvolution�N���X -----------------------------------------
	GMRF��p���ĉ摜�ɃJ�[�l�����t��ݍ��ރN���X (FTMat�N���X���g�p)
	BlurrImage:			�ڂ��摜 (���͗p)
	DeconvImage:		�t��ݍ��݉摜 (�o�͗p)
	KernelFilter:		�J�[�l�� (���͗p)
-----------------------------------------------------------------------*/
class MRF_Deconvolution {
private:
	int x = 0, y = 0, c = 0;
	int index = 0;
public:
	FTMat3D BlurrImage;		// �ڂ��摜
	FTMat3D DeconvImage;	// �t��ݍ��݉摜
	KERNEL KernelFilter;	// �J�[�l��
	Mat GaussianBias;

	double Bias = 1.0e+02;		// HGMRF�p�����[�^
	double Rambda = 1.0e-07;	// GMRF�p�����[�^
	double Alpha = 3.0e-08;
	double Sigma = 70;
	int MAX_Iteration_CG = 106;		// CG�@�̍ő唽����
	double ERROR_END_CG = 1.0e-04;	// CG�@�̎�������l

	MRF_Deconvolution();										// ������
	void deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel);		// �t��ݍ���(GMRF)
	void deconvolved_H(Mat& Blurr, Mat& Output, KERNEL& Kernel);	// �t��ݍ���(HGMRF)

	void settingParameter(double InputRambda, double InputAlpha, double InputSigma, int Input_MAX_Iteration_CG, double Input_ERROR_END_CG);	// �p�����[�^�̐ݒ�
	void settingParameter_H(double InputBias, double InputRambda, double InputAlpha, double InputSigma, int Input_MAX_Iteration_CG, double Input_ERROR_END_CG);
	void Evaluatuion(Mat& True);								// �摜��r�]��
};
MRF_Deconvolution::MRF_Deconvolution() {
	BlurrImage = FTMat3D();
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
	GaussianBias = Mat::zeros(Image_src.size(), CV_64F);
}
void MRF_Deconvolution::deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	/* �O���� */
	// FTMat�N���X�ɕϊ�
	BlurrImage = FTMat3D(Blurr);
	KernelFilter = KERNEL();
	Kernel.copyTo(KernelFilter);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = Blurr.rows + KernelFilter.rows;
	int Nplus = Blurr.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	BlurrImage.DFT();
	KernelFilter.DFT();

	/* �A��1��������Ax=b�����߂� */
	FTMat Matrix_A = FTMat();
	FTMat3D Matrix_Ax = FTMat3D();
	FTMat3D Matrix_b = FTMat3D();

	Mat abs_pow_KernelFilter, dft_b_sub[3];
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2�����x�N�g���̑傫���̂Q��
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_b_sub[c], 0, true);	// ���f����
	}
	Matrix_A = FTMat(abs_pow_KernelFilter, 2);
	Matrix_b = FTMat3D(dft_b_sub[0], dft_b_sub[1], dft_b_sub[2]);
	// inverseDFT�ϊ�
	Matrix_A.iDFT();
	Matrix_b.iDFT();

	double A_calc, b_calc;
	int adject, index2;
	//#pragma omp parallel for private(c)
	for (y = 0; y < Msize; y++) {
		for (x = 0; x < Nsize; x++) {
			index = y * Nsize + x;
			A_calc = Matrix_A.ImgVec.at<double>(0, index);
			A_calc /= Sigma;
			adject = 0;
			if ((x - 1) > 0) { adject++; }
			if ((x + 1) < Nsize) { adject++; }
			if ((y - 1) > 0) { adject++; }
			if ((y + 1) < Msize) { adject++; }
			A_calc += (double)(Rambda + (double)adject * Alpha);
			Matrix_A.ImgVec.at<double>(0, index) = A_calc;

			for (c = 0; c < 3; c++) {
				b_calc = Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index);
				b_calc /= Sigma;

				index2 = y * Nsize + (x - 1);
				if ((x - 1) > 0) {
					index2 = y * Nsize + (x - 1);
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((x + 1) < Nsize) {
					index2 = y * Nsize + (x + 1);
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((y - 1) > 0) {
					index2 = (y - 1) * Nsize + x;
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((y + 1) < Msize) {
					index2 = (y + 1) * Nsize + x;
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) = b_calc;
			}
		}
	}
	// DFT�ϊ�
	Matrix_A.DFT();

	Mat dft_Ax[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, Matrix_A.dft_ImgVec, dft_Ax[c], 0, false);
	}
	Matrix_Ax = FTMat3D(dft_Ax[0], dft_Ax[1], dft_Ax[2]);
	// inverseDFT�ϊ�
	Matrix_Ax.iDFT();
	// 2�����x�N�g���ɕϊ�
	Matrix_Ax.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);
	Matrix_b.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);


	/* Conjugate Gradient (CG) method */
	// �����l�ݒ�
	FTMat3D NextX, LastX;
	NextX = FTMat3D(Blurr);
	LastX = FTMat3D(Blurr);
	NextX.settingAverageColor(BlurrImage);
	LastX.settingAverageColor(BlurrImage);
	FTMat3D Residual, P_base;
	Mat Residual_tmp[3], Residual_tmp3D;
	for (c = 0; c < 3; c++) {
		Residual_tmp[c] = Matrix_b.FT_Mat[c].ImgMat - Matrix_Ax.FT_Mat[c].ImgMat;
	}
	merge(Residual_tmp, 3, Residual_tmp3D);
	Residual = FTMat3D(Residual_tmp3D);
	P_base = FTMat3D(Residual_tmp3D);
	FTMat3D AP = FTMat3D();
	/* �����@�Ōv�Z */
	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < MAX_Iteration_CG; i_number++) {
		// Calculate AP
		multi_Matrix_Vector_3D(Nsize, Msize, Residual, Matrix_A, AP);

		// Calculate ALPHA
		ALPHA = 0.0;
		double Numerator, Denominator;
		for (c = 0; c < 3; c++) {
			//Numerator = multi_vector(P_base.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);		// �x�N�g���̓���
			Numerator = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator = multi_vector(P_base.FT_Mat[c].ImgMat, AP.FT_Mat[c].ImgMat);
			ALPHA += (double)(Numerator / Denominator);
		}
		ALPHA /= 3.0;

		// Calculate x
		for (c = 0; c < 3; c++) {
			NextX.FT_Mat[c].ImgMat = LastX.FT_Mat[c].ImgMat + ALPHA * P_base.FT_Mat[c].ImgMat;
			/*NextX.FT_Mat[c].ImgMat += NextX.AverageColor[c];
			normalize(NextX[c], NextX[c], 0, 1, NORM_MINMAX);
			NextX.FT_Mat[c].ImgMat -= NextX.AverageColor[c];*/
		}

		// Calculate Residual
		Mat Residual_before[3];
		for (c = 0; c < 3; c++) {
			Residual.FT_Mat[c].ImgMat.copyTo(Residual_before[c]);
			Residual.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat - ALPHA * AP.FT_Mat[c].ImgMat;
		}

		energy = 0.0;
		for (c = 0; c < 3; c++) {
			energy += (double)norm(Residual.FT_Mat[c].ImgMat, NORM_L2);
		}
		//energy /= (double)((double)Residual.FT_Mat[c].ImgMat.cols * (double)Residual.FT_Mat[c].ImgMat.rows * 3.0);
		cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
		if (energy < ERROR_END_CG || i_number == MAX_Iteration_CG - 1) {
			//cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2, Denominator2;
		for (c = 0; c < 3; c++) {
			Numerator2 = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator2 = multi_vector(Residual_before[c], Residual_before[c]);
			BETA += (double)(Numerator2 / Denominator2);
		}
		BETA /= 3.0;

		// Calculate P_base
		for (c = 0; c < 3; c++) {
			P_base.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat + BETA * P_base.FT_Mat[c].ImgMat;
		}

		NextX.copyTo(LastX);
	}
	NextX.copyTo(DeconvImage);
	DeconvImage.settingAverageColor(BlurrImage);
	DeconvImage.output(Output);

	// �������̉��
	NextX = FTMat3D();
	LastX = FTMat3D();
	Residual = FTMat3D();
	P_base = FTMat3D();
	AP = FTMat3D();

	/* MRF�p�����[�^�\�� */
	cout << "--- MRF�p�����[�^ --------------------------" << endl;
	cout << " Rambda = " << Rambda << endl;
	cout << " Alpha  = " << Alpha << endl;
	cout << " Sigma  = " << Sigma << endl;
	cout << " �ő唽���� : " << MAX_Iteration_CG << endl;
	cout << " ��������l   : " << ERROR_END_CG << endl;
	cout << "--------------------------------------------" << endl;
	cout << endl;
}
void MRF_Deconvolution::deconvolved_H(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	// Baiasu Parameter
	int HALF_MAX_DATA = (int)(MAX_DATA / 2.0);
	GaussianBias = Mat::zeros(Blurr.size(), CV_64F);
	//int Bx = 0, By = 0;
	//for (int i_B = - HALF_MAX_DATA; i_B < (MAX_DATA - HALF_MAX_DATA); i_B++) {
	//	double B_number = (double)(1.0 / sqrt(2.0 * M_PI * Bias * Bias)) * (double)exp(-(i_B - Bias) * (i_B - Bias) / (2.0 * Bias * Bias));		// �K�E�X�m�C�Y���z
	//	GaussianBias.at<double>(By, Bx) = (double)B_number;
	//	//cout << "  i_B = " << i_B << " : B_number = " << B_number << " = " << (double)GaussianBias.at<double>(By, Bx) << endl;
	//	/*cout << "   " << (double)(1.0 / sqrt(2.0 * M_PI * Bias * Bias)) << endl;
	//	cout << "   " << (double)(-(i_B - Bias) * (i_B - Bias) / (2.0 * Bias * Bias)) << endl;
	//	cout << "   " << (double)exp(-(i_B - Bias) * (i_B - Bias) / (2.0 * Bias * Bias)) << endl;*/
	//	Bx++;
	//	if (Bx >= Blurr.cols) { Bx = 0; }
	//	By++;
	//	if (By >= Blurr.rows) { By = 0; }
	//}
	////cout << "GaussianBias : " << GaussianBias << endl;

	/* �O���� */
	// FTMat�N���X�ɕϊ�
	BlurrImage = FTMat3D(Blurr);
	KernelFilter = KERNEL();
	Kernel.copyTo(KernelFilter);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = Blurr.rows + KernelFilter.rows;
	int Nplus = Blurr.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	BlurrImage.DFT();
	KernelFilter.DFT();

	/* �A��1��������Ax=b�����߂� */
	FTMat Matrix_A = FTMat();
	FTMat3D Matrix_Ax = FTMat3D();
	FTMat3D Matrix_b = FTMat3D();

	Mat abs_pow_KernelFilter, dft_b_sub[3];
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2�����x�N�g���̑傫���̂Q��
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_b_sub[c], 0, true);	// ���f����
	}
	Matrix_A = FTMat(abs_pow_KernelFilter, 2);
	Matrix_b = FTMat3D(dft_b_sub[0], dft_b_sub[1], dft_b_sub[2]);
	// inverseDFT�ϊ�
	Matrix_A.iDFT();
	Matrix_b.iDFT();

	double A_calc, b_calc;
	int adject, index2;
	//#pragma omp parallel for private(c)
	for (y = 0; y < Msize; y++) {
		for (x = 0; x < Nsize; x++) {
			index = y * Nsize + x;
			A_calc = Matrix_A.ImgVec.at<double>(0, index);
			A_calc /= Sigma;
			adject = 0;
			if ((x - 1) > 0) { adject++; }
			if ((x + 1) < Nsize) { adject++; }
			if ((y - 1) > 0) { adject++; }
			if ((y + 1) < Msize) { adject++; }
			A_calc += (double)(Rambda + (double)adject * Alpha);
			Matrix_A.ImgVec.at<double>(0, index) = A_calc;

			for (c = 0; c < 3; c++) {
				b_calc = Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index);
				b_calc /= Sigma;

				index2 = y * Nsize + (x - 1);
				if ((x - 1) > 0) {
					index2 = y * Nsize + (x - 1);
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((x + 1) < Nsize) {
					index2 = y * Nsize + (x + 1);
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((y - 1) > 0) {
					index2 = (y - 1) * Nsize + x;
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				if ((y + 1) < Msize) {
					index2 = (y + 1) * Nsize + x;
					b_calc += Alpha * Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index2);
				}
				Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) = b_calc;
				//cout << "  Matrix_b.FT_Mat[c].ImgVec = " << Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) << endl;	// �m�F

				Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) += (double)Bias;
				//Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) += (double)GaussianBias.at<double>(y, x);
				//cout << "                           => " << Matrix_b.FT_Mat[c].ImgVec.at<double>(0, index) << endl;	// �m�F
			}
		}
	}
	// DFT�ϊ�
	Matrix_A.DFT();

	Mat dft_Ax[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, Matrix_A.dft_ImgVec, dft_Ax[c], 0, false);
	}
	Matrix_Ax = FTMat3D(dft_Ax[0], dft_Ax[1], dft_Ax[2]);
	// inverseDFT�ϊ�
	Matrix_Ax.iDFT();
	// 2�����x�N�g���ɕϊ�
	Matrix_Ax.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);
	Matrix_b.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);


	/* Conjugate Gradient (CG) method */
	// �����l�ݒ�
	FTMat3D NextX, LastX;
	NextX = FTMat3D(Blurr);
	LastX = FTMat3D(Blurr);
	NextX.settingAverageColor(BlurrImage);
	LastX.settingAverageColor(BlurrImage);
	FTMat3D Residual, P_base;
	Mat Residual_tmp[3], Residual_tmp3D;
	for (c = 0; c < 3; c++) {
		Residual_tmp[c] = Matrix_b.FT_Mat[c].ImgMat - Matrix_Ax.FT_Mat[c].ImgMat;
	}
	merge(Residual_tmp, 3, Residual_tmp3D);
	Residual = FTMat3D(Residual_tmp3D);
	P_base = FTMat3D(Residual_tmp3D);
	FTMat3D AP = FTMat3D();
	/* �����@�Ōv�Z */
	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < MAX_Iteration_CG; i_number++) {
		// Calculate AP
		multi_Matrix_Vector_3D(Nsize, Msize, Residual, Matrix_A, AP);

		// Calculate ALPHA
		ALPHA = 0.0;
		double Numerator, Denominator;
		for (c = 0; c < 3; c++) {
			//Numerator = multi_vector(P_base.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);		// �x�N�g���̓���
			Numerator = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator = multi_vector(P_base.FT_Mat[c].ImgMat, AP.FT_Mat[c].ImgMat);
			ALPHA += (double)(Numerator / Denominator);
		}
		ALPHA /= 3.0;

		// Calculate x
		for (c = 0; c < 3; c++) {
			NextX.FT_Mat[c].ImgMat = LastX.FT_Mat[c].ImgMat + ALPHA * P_base.FT_Mat[c].ImgMat;
			/*NextX.FT_Mat[c].ImgMat += NextX.AverageColor[c];
			normalize(NextX[c], NextX[c], 0, 1, NORM_MINMAX);
			NextX.FT_Mat[c].ImgMat -= NextX.AverageColor[c];*/
		}

		// Calculate Residual
		Mat Residual_before[3];
		for (c = 0; c < 3; c++) {
			Residual.FT_Mat[c].ImgMat.copyTo(Residual_before[c]);
			Residual.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat - ALPHA * AP.FT_Mat[c].ImgMat;
		}

		energy = 0.0;
		for (c = 0; c < 3; c++) {
			energy += (double)norm(Residual.FT_Mat[c].ImgMat, NORM_L2);
		}
		//energy /= (double)((double)Residual.FT_Mat[c].ImgMat.cols * (double)Residual.FT_Mat[c].ImgMat.rows * 3.0);
		cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
		if (energy < ERROR_END_CG || i_number == MAX_Iteration_CG - 1) {
			//cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2, Denominator2;
		for (c = 0; c < 3; c++) {
			Numerator2 = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator2 = multi_vector(Residual_before[c], Residual_before[c]);
			BETA += (double)(Numerator2 / Denominator2);
		}
		BETA /= 3.0;

		// Calculate P_base
		for (c = 0; c < 3; c++) {
			P_base.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat + BETA * P_base.FT_Mat[c].ImgMat;
		}

		NextX.copyTo(LastX);
	}
	NextX.copyTo(DeconvImage);
	DeconvImage.settingAverageColor(BlurrImage);
	DeconvImage.output(Output);

	// �������̉��
	NextX = FTMat3D();
	LastX = FTMat3D();
	Residual = FTMat3D();
	P_base = FTMat3D();
	AP = FTMat3D();

	/* MRF�p�����[�^�\�� */
	cout << "--- MRF�p�����[�^(HGMRF) -------------------" << endl;
	cout << " Bias   = " << Bias << endl;
	cout << " Rambda = " << Rambda << endl;
	cout << " Alpha  = " << Alpha << endl;
	cout << " Sigma  = " << Sigma << endl;
	cout << " �ő唽���� : " << MAX_Iteration_CG << endl;
	cout << " ��������l   : " << ERROR_END_CG << endl;
	cout << "--------------------------------------------" << endl;
	cout << endl;
}
void MRF_Deconvolution::settingParameter(double InputRambda, double InputAlpha, double InputSigma, int Input_MAX_Iteration_CG, double Input_ERROR_END_CG) {
	Rambda = InputRambda;	// GMRF�p�����[�^
	Alpha = InputAlpha;
	Sigma = InputSigma;
	MAX_Iteration_CG = Input_MAX_Iteration_CG;	// CG�@�̍ő唽����
	ERROR_END_CG = Input_ERROR_END_CG;			// CG�@�̎�������l
}
void MRF_Deconvolution::settingParameter_H(double InputBias, double InputRambda, double InputAlpha, double InputSigma, int Input_MAX_Iteration_CG, double Input_ERROR_END_CG) {
	Bias = InputBias;		// HGMRF�p�����[�^
	Rambda = InputRambda;	// GMRF�p�����[�^
	Alpha = InputAlpha;
	Sigma = InputSigma;
	MAX_Iteration_CG = Input_MAX_Iteration_CG;	// CG�@�̍ő唽����
	ERROR_END_CG = Input_ERROR_END_CG;			// CG�@�̎�������l
}
void MRF_Deconvolution::Evaluatuion(Mat& True) {
	Mat Deconvolved_Img;
	DeconvImage.output(Deconvolved_Img);

	cout << "�ڂ������摜(MRF-based) �� �^�摜" << endl;
	Evaluation_MSE_PSNR_SSIM(True, Deconvolved_Img);
}


/* �֐� */

/*--- multi_Matrix_Vector_3D() ----------------------------------------
	2�����x�N�g����1�����x�N�g���̏�Z���v�Z����N���X
	fft_x:		�t�[���G�ϊ��� x�����T�C�Y�i���͗p�j
	fft_y:		�t�[���G�ϊ��� y�����T�C�Y�i���͗p�j
	InputImg:				1�����x�N�g���i���͗p�j
	dft_InputTransVector:	2�����x�N�g����DFT�i���͗p�j
	OutputImg:				1�����x�N�g���i�o�͗p�j
------------------------------------------------------------------------*/
void multi_Matrix_Vector_3D(int fft_x, int fft_y, FTMat3D& InputImg, FTMat& dft_InputTransVector, FTMat3D& OutputImg) {
	int c;
	int Msize = fft_y;
	int Nsize = fft_x;
	OutputImg = FTMat3D();

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	InputImg.toVector(1, 0, 1, Nsize, Msize);
	//dft_InputTransVector.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	InputImg.DFT();
	//dft_InputTransVector.DFT();
	// 2�����x�N�g����1�����x�N�g���̏�Z�����߂�
	Mat dft_NewImage[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(InputImg.FT_Mat[c].dft_ImgVec, dft_InputTransVector.dft_ImgVec, dft_NewImage[c], 0, false);
	}

	OutputImg = FTMat3D(dft_NewImage[0], dft_NewImage[1], dft_NewImage[2]);
	OutputImg.settingAverageColor(InputImg);
	// inverseDFT�ϊ�
	OutputImg.iDFT();
	// 2�����x�N�g���ɕϊ�
	OutputImg.toMatrix(1, 0, InputImg.FT_Mat[0].ImgMat.cols, InputImg.FT_Mat[0].ImgMat.rows, Nsize, Msize);
}

/*--- multi_vector() --------------------------------------------------
	1�����x�N�g���̏�Z(����)���v�Z����N���X
	srcImg:		1�����x�N�g��A�i���͗p�j
	srcImg2:	1�����x�N�g��B�i���͗p�j
------------------------------------------------------------------------*/
double multi_vector(Mat& srcImg, Mat& srcImg2) {
	double Result = 0.0;
	int x, y;
	double in, in2, out;

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_vector() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 1) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<double>(y, x);
				in2 = srcImg2.at<double>(y, x);
				out = in * in2;
				Result += out;
			}
		}
	}
	else { cout << "ERROR! multi_vector() : Can't translate because of wrong channels." << endl; }

	return Result;
}


#endif