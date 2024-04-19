from settings import loader_settings
import medpy.io
from medpy.io import load,save
import os, pathlib
import subprocess
import numpy as np
import shutil
import SimpleITK as sitk
from pathlib import Path
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
class Seg():
	def __init__(self):
		return
	
	def process(self):
		
		inp_path = loader_settings['InputPath']  # Path for the input
		out_path = loader_settings['OutputPath']  # Path for the output
		processed_inp_path = 'nnUNet/nnUNet_data/'
		
		file_list = os.listdir(inp_path)  # List of files in the input
		#file_list = [os.path.join(inp_path, f) for f in file_list]
		#print('before copying',file_list)
		mapping_filename = {}
		print('No. of files',len(file_list))
		print('copy to nnunet format')
		for file_name  in file_list:
			print(file_name)
			basename = file_name.split('.')[0]
			if '.nii.gz' in file_name:
				shutil.copy(inp_path+file_name,processed_inp_path + basename +'_0000.nii.gz')
			else:
				file_sitk_img = sitk.ReadImage(inp_path+file_name)
				print(file_sitk_img.GetSize())
				mapping_filename[file_name] = [file_sitk_img.GetOrigin(),file_sitk_img.GetSpacing(),file_sitk_img.GetDirection(),file_sitk_img.GetSize()]
				sitk.WriteImage(file_sitk_img, os.path.join(processed_inp_path, basename + '_0000.nii.gz'))
		print('Done')
		file_list = os.listdir(inp_path)  # List of files in the input
		#print('after copying',file_list)		
		
		print('Number of processed files', len(os.listdir(processed_inp_path)))
		#print(os.listdir(processed_inp_path))
		
		trainer_model = ['nnUNetTrainer__nnUNetPlans__3d_fullres','nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres','nnUNetTrainerDA5__nnUNetPlans__3d_fullres']
		folds = ['fold_0','fold_1','fold_2','fold_3','fold_4']
		output_folder_paths = []
		for model in trainer_model:
			for fold in folds:
				subprocess.run(['mkdir',model+'__'+fold + '__pred'])
				output_folder_paths.append(model +'__'+fold +'__pred')
				#print('mkdir',model+'__'+fold + '__pred')
		
		
		file_list = os.listdir(processed_inp_path)
		for file_name in file_list:
			sample = sitk.ReadImage(processed_inp_path+file_name)
			print(sample.GetSize())
		
		for path in output_folder_paths:
				subprocess.run(['nnUNetv2_predict','-i',processed_inp_path,'-o', path +'/', '-d', '001', '-p' , path.split('__')[1],'-tr' , path.split('__')[0] , '-c' , '3d_fullres','-f' ,path.split('__')[-2].split('_')[-1], '-chk' , 'checkpoint_best.pth', '--save_probabilities', '--disable_progress_bar' ]) #,  '-npp', '1' ,'-nps','1'])
		
		#Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
		
			
		file_list = os.listdir(output_folder_paths[0]+'/')
		print('Total files - ',len(file_list))
		file_list = [f for f in file_list if '.nii.gz' in f]
		for f in file_list:
			sample = sitk.ReadImage(output_folder_paths[0]+'/'+f)
			print(sample.GetSize())
		print('Number of predicted files',len(file_list))
		
		ensemble_output = 'nnUNet/nnUNet_predicted/'
		run_ensemble = ['nnUNetv2_ensemble', '-i']
		input_folder_string = ''
		for path in output_folder_paths:
			run_ensemble.append(path + '/')
		run_ensemble.append('-o')
		run_ensemble.append(ensemble_output)
		print('ensemble command',run_ensemble)
		subprocess.run(run_ensemble)
		
		file_name_list = os.listdir(inp_path)
		for file_name in file_name_list:
			base_name = file_name.split('.')[0]
			sitk_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(ensemble_output+base_name + '.nii.gz')))
			print(sitk_image.GetSize())
			sitk_image.SetOrigin(mapping_filename[file_name][0])
			sitk_image.SetSpacing(mapping_filename[file_name][1])
			sitk_image.SetDirection(mapping_filename[file_name][2])
			sitk.WriteImage(sitk_image, os.path.join(out_path, file_name), True)
			print('saving .mha')
		'''
		file_name_list = os.listdir(inp_path)
		#print('file_name_list',file_name_list)
		print('Loading files for ensemble')		
		for file_name in file_name_list:
			basename = file_name.split('.')[0]
			final_ans = np.zeros(shape=mapping_filename[file_name][3])
			print('ensembling',file_name)
			for folder in output_folder_paths:
				print(folder)
				data,hdr = load(folder  + '/' + basename + '.nii.gz')
				print(data.shape,final_ans.shape)
				final_ans += data
			final_ans = final_ans/len(output_folder_paths)
			thres_ans = np.zeros(shape=mapping_filename[file_name][3])
			thres_ans[final_ans > 0.5] =  1.0
			if '.nii.gz' in file_name: # suffix is .nii.gz
				out_name = out_path+file_name
				save(thres_ans,out_name,hdr)
				print('saving nii.gz')
			else:
				sitk_image = sitk.GetImageFromArray(thres_ans)
				sitk_image.SetOrigin(mapping_filename[file_name][0])
				sitk_image.SetSpacing(mapping_filename[file_name][1])
				sitk_image.SetDirection(mapping_filename[file_name][2])
				sitk.WriteImage(sitk_image, os.path.join(out_path, file_name), True)
				print('saving .mha')
		
		for file_name in file_list:
			final_ans = np.zeros(shape=(197,233,189))
			for folder in output_folder_paths:
				data,hdr = load(folder  + '/' + file_name)
				print(data.shape,final_ans.shape)
				final_ans += data
			final_ans = final_ans/len(output_folder_paths)
			thres_ans = np.zeros(shape=(197,233,189))
			thres_ans[final_ans > 0.5] =  1.0
			#ADD POSTPROCESSING
			out_name = out_path+file_name.replace('_0000','')
			save(thres_ans,out_name,hdr)
		'''
		
		
		file_list = os.listdir(out_path)  # List of files in the input
		#file_list = [os.path.join(out_path, f) for f in file_list]
		print('No. of output files',len(file_list))
		for f in file_list:
			sample = sitk.ReadImage(out_path+f)
			print(sample.GetSize())

		return


if __name__ == "__main__":
	pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
	Seg().process()
