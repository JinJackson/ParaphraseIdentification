checkpoint = -1
round = -1
checkpoint_dir_name = r'D:\Dev\pythonspace\pycharmWorkspace\NLPtasks\ParaphraseIdentification\result\bert\baseline\LCQMC\checkpoint-2-1'
checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
round = max(round, int(checkpoint_dir_name.split('/')[-1].split('-')[-1]))
print(checkpoint, round)