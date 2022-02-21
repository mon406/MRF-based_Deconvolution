#ifndef __INCLUDED_H_Convolution__
#define __INCLUDED_H_Convolution__

#include "main.h"
#include "Fourier_Transform_Mat.h"
#include "Calculate_Complex_Number.h"
#include "Image_Evaluation.h"			// �摜�̕]��(MSE, PSNR, SSIM)


/*--- Convolution�N���X -----------------------------------------------
	DFT��p���ĉ摜�ɃJ�[�l������ݍ��ރN���X (FTMat�N���X���g�p)
	TrueImage:			�^�摜 (���͗p)
	ConvImage:			��ݍ��݉摜 (�o�͗p)
	KernelFilter:		�J�[�l�� (�o�͗p)
	KernelFastOrder:	�J�[�l���w��l (���͗p)
-----------------------------------------------------------------------*/
class Convolution {
private:
	int c = 0;
public:
	FTMat3D TrueImage;		// �^�摜 (�J���[)
	FTMat3D ConvImage;		// ��ݍ��݉摜
	KERNEL KernelFilter;	// �J�[�l��

	Convolution();																	// ������
	void convolved(Mat& True, Mat& Output, KERNEL& Kernel, int KernelFastOrder);	// �J�[�l���̏�ݍ���(�ڂ��摜�쐬)
	void Evaluatuion();																// �摜��r�]��
};
Convolution::Convolution() {
	TrueImage = FTMat3D();
	ConvImage = FTMat3D();
	KernelFilter = KERNEL();
}
void Convolution::convolved(Mat& True, Mat& Output, KERNEL& Kernel, int KernelFastOrder) {
	/* �O���� */
	// FTMat�N���X�ɕϊ�
	TrueImage = FTMat3D(True);
	ConvImage = FTMat3D();
	KernelFilter = KERNEL(KernelFastOrder);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = True.rows + KernelFilter.rows;
	int Nplus = True.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")";	// �m�F
	cout << " : (" << TrueImage.FT_Mat[0].ImgMat.cols << "+" << KernelFilter.cols << ", " << TrueImage.FT_Mat[0].ImgMat.rows << "+" << KernelFilter.rows << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	TrueImage.DFT();
	KernelFilter.DFT();

	/* �ڂ��摜�����߂� */
	Mat dft_ConvolvedImg[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(TrueImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_ConvolvedImg[c], 0, false);
	}

	/* �o�� */
	ConvImage = FTMat3D(dft_ConvolvedImg[0], dft_ConvolvedImg[1], dft_ConvolvedImg[2]);
	ConvImage.settingAverageColor(TrueImage);
	// inverseDFT�ϊ�
	ConvImage.iDFT();
	// 2�����x�N�g���ɕϊ�
	ConvImage.toMatrix(1, 0, True.cols, True.rows, Nsize, Msize);


	/* �t���m�F�p */
	//// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	//ConvImage.toVector(1, 0, 1, Nsize, Msize);
	//// DFT�ϊ�
	//ConvImage.DFT();
	//// �t��ݍ���
	//Mat dft_DeconvolvedImg[3];
	//Mat abs_pow_KernelFilter, denom_KernelFilter;
	//abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2�����x�N�g���̑傫���̂Q��
	//reciprocal_complex_Mat(abs_pow_KernelFilter, denom_KernelFilter);		// 2�����x�N�g���̋t��
	//for (c = 0; c < 3; c++) {
	//	mulSpectrums(ConvImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_DeconvolvedImg[c], 0, true);	// ���f����
	//	mulSpectrums(dft_DeconvolvedImg[c], denom_KernelFilter, dft_DeconvolvedImg[c], 0, false);
	//}
	//ConvImage = FTMat3D();
	//ConvImage = FTMat3D(dft_DeconvolvedImg[0], dft_DeconvolvedImg[1], dft_DeconvolvedImg[2]);
	//ConvImage.settingAverageColor(TrueImage);
	//// inverseDFT�ϊ�
	//ConvImage.iDFT();
	//// 2�����x�N�g���ɕϊ�
	//ConvImage.toMatrix(1, 0, True.cols, True.rows, Nsize, Msize);


	KernelFilter.copyTo(Kernel);
	ConvImage.output(Output);
}
void Convolution::Evaluatuion() {
	Mat True_Img, Convolved_Img;
	TrueImage.output(True_Img);
	ConvImage.output(Convolved_Img);

	cout << "�ڂ��摜 �� �^�摜(�K�E�X�m�C�Y�Ȃ�)" << endl;
	Evaluation_MSE_PSNR_SSIM(True_Img, Convolved_Img);
}


/*--- Deconvolution�N���X ---------------------------------------------
	DFT��p���ĉ摜�ɃJ�[�l�����t��ݍ��݂���N���X (FTMat�N���X���g�p)
	�C���o�[�X�E�t�B���^�[�ɂ��f�R���{�����[�V����
	BlurrImage:			�ڂ��摜 (���͗p)
	DeconvImage:		�t��ݍ��݉摜 (�o�͗p)
	KernelFilter:		�J�[�l�� (���͗p)
-----------------------------------------------------------------------*/
class Deconvolution {
private:
	int c = 0;
public:
	FTMat3D BlurrImage;		// �ڂ��摜
	FTMat3D DeconvImage;	// �t��ݍ��݉摜
	KERNEL KernelFilter;	// �J�[�l��

	Deconvolution();											// ������
	void deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel);	// �P��(���_�I)�t��ݍ���
	void Evaluatuion(Mat& True);								// �摜��r�]��
};
Deconvolution::Deconvolution() {
	BlurrImage = FTMat3D();
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
}
void Deconvolution::deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	/* �O���� */
	// FTMat�N���X�ɕϊ�
	BlurrImage = FTMat3D(Blurr);
	DeconvImage = FTMat3D();
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

	/* �ڂ������摜(�^�̌��摜)�����߂� */
	Mat dft_DeconvolvedImg[3];
	Mat abs_pow_KernelFilter, denom_KernelFilter;
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2�����x�N�g���̑傫���̂Q��
	reciprocal_complex_Mat(abs_pow_KernelFilter, denom_KernelFilter);		// 2�����x�N�g���̋t��
	//reciprocal_complex_Mat(KernelFilter.dft_ImgVec, denom_KernelFilter);		// �J�[�l���̋t��
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_DeconvolvedImg[c], 0, true);	// ���f����
		mulSpectrums(dft_DeconvolvedImg[c], denom_KernelFilter, dft_DeconvolvedImg[c], 0, false);

		//mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, denom_KernelFilter, dft_DeconvolvedImg[c], 0, false);
	}

	/* �o�� */
	DeconvImage = FTMat3D(dft_DeconvolvedImg[0], dft_DeconvolvedImg[1], dft_DeconvolvedImg[2]);
	DeconvImage.settingAverageColor(BlurrImage);
	// inverseDFT�ϊ�
	DeconvImage.iDFT();
	// 2�����x�N�g���ɕϊ�
	DeconvImage.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);

	DeconvImage.output(Output);
}
void Deconvolution::Evaluatuion(Mat& True) {
	Mat Deconvolved_Img;
	DeconvImage.output(Deconvolved_Img);

	cout << "�ڂ������摜 �� �^�摜" << endl;
	Evaluation_MSE_PSNR_SSIM(True, Deconvolved_Img);
}

/*--- Deconvolution_WF�N���X ------------------------------------------
	Deconvolution�N���X���p�������t��ݍ��݃N���X (FTMat�N���X���g�p)
	�E�B�[�i�E�t�B���^�[�ɂ��f�R���{�����[�V����
	TrueImage:			�^�摜 (���͗p)
	BlurrImage:			�ڂ��摜 (���͗p)
	DeconvImage:		�t��ݍ��݉摜 (�o�͗p)
	KernelFilter:		�J�[�l�� (���͗p)
-----------------------------------------------------------------------*/
class Deconvolution_WF : public Deconvolution {
private:
	int x = 0, y = 0, c = 0;
	int index = 0;
public:
	FTMat3D TrueImage;		// �^�摜 (�J���[)
	//FTMat3D BlurrImage;	// �ڂ��摜
	//FTMat3D DeconvImage;	// �t��ݍ��݉摜
	//KERNEL KernelFilter;	// �J�[�l��
	Mat dft_TrueConstant;		// �E�B�[�i�E�t�B���^�[�̐^�̌W��

	Deconvolution_WF();													// ������
	void calcWienerFilterConstant(Convolution& InputData, Mat& Blurr);	// FFT�̌덷���^�̌W�������߂�
	void calcWienerFilterConstant2(Convolution& InputData, Mat& Blurr);	// �m�C�Y�Ɛ^�摜�̃p���[�X�y�N�g�����^�̌W�������߂�
	void calcWienerFilterConstant3(Convolution& InputData, Mat& Blurr);
	void deconvolved_WF(Mat& Blurr, Mat& Output, KERNEL& Kernel);		// �t��ݍ��� (�E�B�[�i�E�t�B���^�[)
};
Deconvolution_WF::Deconvolution_WF() {
	TrueImage = FTMat3D();
	BlurrImage = FTMat3D();
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
	dft_TrueConstant.release();
}
void Deconvolution_WF::calcWienerFilterConstant(Convolution& InputData, Mat& Blurr) {
	/* Convolution��FTMat�N���X��p����*/
	BlurrImage = FTMat3D(Blurr);
	InputData.TrueImage.copyTo(TrueImage);
	InputData.KernelFilter.copyTo(KernelFilter);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = TrueImage.FT_Mat[0].ImgMat.rows + KernelFilter.rows;
	int Nplus = TrueImage.FT_Mat[0].ImgMat.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	BlurrImage.DFT();
	TrueImage.DFT();
	KernelFilter.DFT();

	/* �^�̌W�����v�Z */
	Mat dft_TrueConstant_RGB[3];
	Mat denom_dft_TrueImage[3];
	Mat dft_DeconvolvedImg_tmp1[3], dft_DeconvolvedImg_tmp2[3];
	Mat abs_pow_KernelFilter;
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2�����x�N�g���̑傫���̂Q��
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_DeconvolvedImg_tmp1[c], 0, true);	// ���f����
		reciprocal_complex_Mat(TrueImage.FT_Mat[c].dft_ImgVec, denom_dft_TrueImage[c]);									// 2�����x�N�g���̋t��
		mulSpectrums(dft_DeconvolvedImg_tmp1[c], denom_dft_TrueImage[c], dft_DeconvolvedImg_tmp2[c], 0, false);

		dft_TrueConstant_RGB[c] = Mat::zeros(1, Nsize * Msize, CV_64FC2);
		dft_TrueConstant_RGB[c] = dft_DeconvolvedImg_tmp2[c] - abs_pow_KernelFilter;
	}

	// �^�̌W����Mat�m�F
	dft_TrueConstant = Mat::zeros(1, Nsize * Msize, CV_64FC2);
	Vec2d ave_number[3], ave_num;
	Vec2d All_Num = { (double)Nsize * (double)Msize, 0.0 };
	Vec2d Color_Num = { 3.0, 0.0 };
	Vec2d Error = { 0.0, 0.0 };
#pragma omp parallel for private(y, x, index)
	for (c = 0; c < 3; c++) {
		ave_number[c] = { 0.0, 0.0 };
		index = 0;
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				ave_num = dft_TrueConstant_RGB[c].at<Vec2d>(0, index);
				ave_number[c] += ave_num;
				divi_Vec2d(ave_num, ave_num, Color_Num);
				dft_TrueConstant.at<Vec2d>(0, index) += ave_num;
				index++;
			}
		}
		divi_Vec2d(ave_number[c], ave_number[c], All_Num);
		cout << "  " << c << " : " << ave_number[c] << endl;	// �m�F�p

		Error += ave_number[c];
	}
	divi_Vec2d(Error, Error, Color_Num);
	cout << " Error = " << Error << endl;	// �m�F�p
	cout << endl;
	//checkMat_detail(dft_TrueConstant);	// �m�F�p
}
void Deconvolution_WF::calcWienerFilterConstant2(Convolution& InputData, Mat& Blurr) {
	/* Convolution��FTMat�N���X��p����*/
	InputData.TrueImage.copyTo(TrueImage);
	// �m�C�Y�̎Z�o
	Mat ConvImage_original;		// �^�摜 (�K�E�X�m�C�Y�t���O)
	InputData.ConvImage.output(ConvImage_original);
	ConvImage_original.convertTo(ConvImage_original, CV_64FC3);
	Mat ConvImage_noisy;		// �^�摜 (�K�E�X�m�C�Y�t����)
	Blurr.convertTo(ConvImage_noisy, CV_64FC3);
	Mat NoiseImage = Mat::zeros(ConvImage_original.size(), CV_64FC3);	// �m�C�Y
	NoiseImage = ConvImage_noisy - ConvImage_original;
	FTMat3D TrueNoise = FTMat3D(NoiseImage);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = TrueImage.FT_Mat[0].ImgMat.rows + InputData.KernelFilter.rows;
	int Nplus = TrueImage.FT_Mat[0].ImgMat.cols + InputData.KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	TrueNoise.toVector(1, 0, 1, Nsize, Msize);
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	// DFT�ϊ�
	TrueNoise.DFT();
	TrueImage.DFT();

	/* �^�̌W�����v�Z */
	Mat dft_TrueConstant_RGB[3];
	Mat denom_dft_TrueImage[3];
	Mat abs_pow_TrueNoise[3], abs_pow_TrueImage[3], denom_abs_pow_TrueImage[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex_Mat(TrueNoise.FT_Mat[c].dft_ImgVec, abs_pow_TrueNoise[c]);	// 2�����x�N�g���̑傫���̂Q��
		abs_pow_complex_Mat(TrueImage.FT_Mat[c].dft_ImgVec, abs_pow_TrueImage[c]);
		reciprocal_complex_Mat(abs_pow_TrueImage[c], denom_abs_pow_TrueImage[c]);	// 2�����x�N�g���̋t��
		mulSpectrums(abs_pow_TrueNoise[c], denom_abs_pow_TrueImage[c], dft_TrueConstant_RGB[c], 0, false);
	}

	// �^�̌W����Mat�m�F
	dft_TrueConstant = Mat::zeros(1, Nsize * Msize, CV_64FC2);
	Vec2d ave_number[3], ave_num;
	Vec2d All_Num = { (double)Nsize * (double)Msize, 0.0 };
	Vec2d Color_Num = { 3.0, 0.0 };
	Vec2d Error = { 0.0, 0.0 };
#pragma omp parallel for private(y, x, index)
	for (c = 0; c < 3; c++) {
		ave_number[c] = { 0.0, 0.0 };
		index = 0;
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				ave_num = dft_TrueConstant_RGB[c].at<Vec2d>(0, index);
				ave_number[c] += ave_num;
				divi_Vec2d(ave_num, ave_num, Color_Num);
				dft_TrueConstant.at<Vec2d>(0, index) += ave_num;
				index++;
			}
		}
		divi_Vec2d(ave_number[c], ave_number[c], All_Num);
		cout << "  " << c << " : " << ave_number[c] << endl;	// �m�F�p

		Error += ave_number[c];
	}
	divi_Vec2d(Error, Error, Color_Num);
	cout << " Error = " << Error << endl;	// �m�F�p
	cout << endl;
	//checkMat_detail(dft_TrueConstant);	// �m�F�p
}
void Deconvolution_WF::calcWienerFilterConstant3(Convolution& InputData, Mat& Blurr) {
	/* Convolution��FTMat�N���X��p����*/
	InputData.TrueImage.copyTo(TrueImage);
	// �m�C�Y(5�~5�̃K�E�V�A���t�B���^)
	Mat GaussianFilter = (Mat_<double>(5, 5) 		// 5*5
		<< 1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1);
	GaussianFilter /= 256.0;
	FTMat TrueNoise = FTMat(GaussianFilter, 0);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = TrueImage.FT_Mat[0].ImgMat.rows + InputData.KernelFilter.rows;
	int Nplus = TrueImage.FT_Mat[0].ImgMat.cols + InputData.KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F
	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	TrueNoise.toVector(1, 0, 1, Nsize, Msize);
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	// DFT�ϊ�
	TrueNoise.DFT();
	TrueImage.DFT();

	/* �^�̌W�����v�Z */
	Mat dft_TrueConstant_RGB[3];
	Mat denom_dft_TrueImage[3];
	Mat abs_pow_TrueNoise, abs_pow_TrueImage[3], denom_abs_pow_TrueImage[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex_Mat(TrueNoise.dft_ImgVec, abs_pow_TrueNoise);	// 2�����x�N�g���̑傫���̂Q��
		abs_pow_complex_Mat(TrueImage.FT_Mat[c].dft_ImgVec, abs_pow_TrueImage[c]);
		reciprocal_complex_Mat(abs_pow_TrueImage[c], denom_abs_pow_TrueImage[c]);	// 2�����x�N�g���̋t��
		mulSpectrums(abs_pow_TrueNoise, denom_abs_pow_TrueImage[c], dft_TrueConstant_RGB[c], 0, false);
	}

	// �^�̌W����Mat�m�F
	dft_TrueConstant = Mat::zeros(1, Nsize * Msize, CV_64FC2);
	Vec2d ave_number[3], ave_num;
	Vec2d All_Num = { (double)Nsize * (double)Msize, 0.0 };
	Vec2d Color_Num = { 3.0, 0.0 };
	Vec2d Error = { 0.0, 0.0 };
#pragma omp parallel for private(y, x, index)
	for (c = 0; c < 3; c++) {
		ave_number[c] = { 0.0, 0.0 };
		index = 0;
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				ave_num = dft_TrueConstant_RGB[c].at<Vec2d>(0, index);
				ave_number[c] += ave_num;
				divi_Vec2d(ave_num, ave_num, Color_Num);
				dft_TrueConstant.at<Vec2d>(0, index) += ave_num;
				index++;
			}
		}
		divi_Vec2d(ave_number[c], ave_number[c], All_Num);
		cout << "  " << c << " : " << ave_number[c] << endl;	// �m�F�p

		Error += ave_number[c];
	}
	divi_Vec2d(Error, Error, Color_Num);
	cout << " Error = " << Error << endl;	// �m�F�p
	cout << endl;
	//checkMat_detail(dft_TrueConstant);	// �m�F�p
}
void Deconvolution_WF::deconvolved_WF(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	/* �O���� */
	// FTMat�N���X�ɕϊ�
	BlurrImage = FTMat3D(Blurr);
	DeconvImage = FTMat3D();
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

	/* �ڂ������摜(�^�̌��摜)�����߂� */
	Mat dft_DeconvolvedImg[3];
	Mat denom_WienerFilter;
	wiener_filter(KernelFilter.dft_ImgVec, denom_WienerFilter, dft_TrueConstant);	// 2�����x�N�g���̋t��(�E�B�[�i�E�t�B���^�[)
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, denom_WienerFilter, dft_DeconvolvedImg[c], 0, false);
	}

	/* �o�� */
	DeconvImage = FTMat3D(dft_DeconvolvedImg[0], dft_DeconvolvedImg[1], dft_DeconvolvedImg[2]);
	DeconvImage.settingAverageColor(BlurrImage);
	// inverseDFT�ϊ�
	DeconvImage.iDFT();
	// 2�����x�N�g���ɕϊ�
	DeconvImage.toMatrix(1, 0, Blurr.cols, Blurr.rows, Nsize, Msize);

	DeconvImage.output(Output);
}


#endif