Error Log 

## RuntimeError During Model Training

object address : 0x16b765b40  
object refcount : 2  
object type : 0x105d56c30  
object type name: ValueError  
object repr : ValueError('Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])')  
lost sys.stderr

### Cause

This error typically occurs when a batch with only one image is passed through BatchNorm layers. BatchNorm requires more than one sample per channel during training to compute meaningful statistics.

### Solution

Even though your BATCH_SIZE is set to 2, if the number of training samples isn't divisible by 2, the last batch may contain only one sample.

Fix this by setting drop_last=True in your DataLoader:

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

This ensures that the last incomplete batch is dropped during training.

## labelImg Launch Error on macOS

(venv) user@Mac-mini % labelImg
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in "/opt/homebrew/opt/qt@5/plugins/platforms"
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

zsh: abort labelImg

###  Cause

This error means the Qt platform plugin for macOS (cocoa) cannot be found. It usually happens when:

Qt or PyQt5 is not installed correctly.

Environment paths are misconfigured.

labelImg is broken or incompatible.

###  Solutions

Reinstall PyQt5 and labelImg in a fresh environment

pip uninstall pyqt5 labelImg
pip install pyqt5==5.15.9
pip install labelImg==1.8.6

At one point 




