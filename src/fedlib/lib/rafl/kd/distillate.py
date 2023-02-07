
# TODO: define kd in init() function.
from .dml import DML
from .vanilla import VanillaKD


class Distiller():
    def __init__(self, lr=0.01, epochs=5, kd_type='VanillaKD', device='cpu'):
        """


            :param loss_fn (torch.nn.Module):  Calculates loss during distillation
            :param temp (float): Temperature parameter for distillation
            :param distil_weight (float): Weight paramter for distillation loss
            :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
            :param log (bool): True if logging required
            :param logdir (str): Directory for storing logs

        """
        super(Distiller, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.kd_type = kd_type
        self.device = device
        self.teacher = None
        self.student = None
        self.kd = None
        self.teacher_optimizer = None
        self.student_optimizer = None
        self.train_loader = None
        self.test_loader = None


    def joint_kd(self,teacher, student, teacher_optimizer, student_optimizer, train_loader, test_loader, t_save_path=None, s_save_path=None):
        """
            Jointly distillation. Train the teacher and then distillate the student.
            Both teacher and student will be updated.
            :param teacher_model (torch.nn.Module): Teacher model
            :param student_model (torch.nn.Module): Student model
            :param train_loader (torch.utils.data.DataLoader): Dataloader for training
            :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
            :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
            :param optimizer_student (torch.optim.*): Optimizer used for training student
        """
        if self.kd_type == 'VanillaKD':
            kd = VanillaKD(teacher, student, train_loader, test_loader,
                           teacher_optimizer, student_optimizer, device=self.device)

        kd.train_teacher(epochs=self.epochs, plot_losses=False, save_model=True,save_model_pth=t_save_path)
        kd.train_student(epochs=self.epochs, plot_losses=False, save_model=True,save_model_pth=s_save_path)

        tch_acc = kd.evaluate(teacher=True)  # eval student
        stu_acc = kd.evaluate(teacher=False)  # eval student
        return kd.teacher_model, kd.student_model, tch_acc, stu_acc

    def pure_kd(self, teacher, student, teh_optimizer, stu_optimizer, train_loader, test_loader,
                device='cuda',s_save_path=None):
        '''
            Pure distillation. Distillate the teacher's knowledge to student.
            Only student will be updated.
            :param teacher_model (torch.nn.Module): Teacher model
            :param student_model (torch.nn.Module): Student model
            :param train_loader (torch.utils.data.DataLoader): Dataloader for training
            :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
            :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
            :param optimizer_student (torch.optim.*): Optimizer used for training student
        '''


        if self.kd_type == 'VanillaKD':
            kd = VanillaKD(teacher, student, train_loader, test_loader,
                           teh_optimizer, stu_optimizer,lr=self.lr ,device=device)
        lr = kd.train_student(epochs=self.epochs, plot_losses=False, save_model=False)

        stu_acc = kd.evaluate(teacher=False)  # eval student
        return kd.teacher_model, kd.student_model,stu_acc, lr


    #TODO: return stu network acc
    def mutual_kd(self,nets,train_loader, test_loader,optimizers,s_save_path=None):


        kd = DML(nets, train_loader, test_loader, optimizers,lr=self.lr,
                        device=self.device)
        lr = kd.train_students(epochs=self.epochs,plot_losses=False,save_model=False)
        # kd.evaluate()
        return kd.student_cohort, lr

    def eval(self):
        return



# if __name__ == '__main__':
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "mnist_data",
#             train=True,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#             ),
#         ),
#         batch_size=32,
#         shuffle=True,
#     )
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "mnist_data",
#             train=False,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#             ),
#         ),
#         batch_size=32,
#         shuffle=True,
#     )
#
#     kd_agent = Distiller(epochs=2,device='cuda')
#     student_params = [4, 4, 4, 4, 4]
#     teacher_model = ResNet50(student_params, 1, 10)
#     student_model = ResNet18(student_params, 1, 10)
#
#     teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
#     student_optimizer = optim.SGD(student_model.parameters(), 0.01)
#
'''
    def training_setup(self, epochs: int = None, lr: float = None):
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr

    def kd_setup(self):
        if self.kd_type == 'VanillaKD':
            self.kd = VanillaKD(self.teacher, self.student, self.train_loader, self.test_loader,
                                self.teacher_optimizer, self.student_optimizer, device=self.device)

    def kd_setup(self, teacher, student, train_loader, test_loader, teacher_optimizer, student_optimizer):
        if self.kd_type == 'VanillaKD':
            self.kd = VanillaKD(teacher, student, train_loader, test_loader,
                                teacher_optimizer, student_optimizer, device=self.device)

    def optimizer_setup(self, teacher_optimizer=None, student_optimizer=None):
        if teacher_optimizer is not None:
            self.teacher_optimizer = teacher_optimizer
        if student_optimizer is not None:
            self.student_optimizer = student_optimizer
        self.kd_setup()

    def net_setup(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def loader_setup(self, train_loader=None, test_loader=None):
        if test_loader is not None:
            self.test_loader = test_loader
        if train_loader is not None:
            self.train_loader = train_loader
        self.kd_setup()

    def kd_reset(self):
        self.teacher_optimizer = None
        self.student_optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.kd = None

'''
#     teacher_model, student_model,tch_acc, stu_acc = kd_agent.joint_kd(teacher_model, student_model, teacher_optimizer, student_optimizer,train_loader, test_loader,'models/teacher.pth','models/stu.pth')
