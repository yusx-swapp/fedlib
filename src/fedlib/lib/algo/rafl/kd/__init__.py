from .dml import DML
from .vanilla import VanillaKD
from torch import nn



def client_dml(nets, train_loader, optimizers, lr_schedulers,epochs, device, \
                loss_fn=nn.MSELoss(), plot_losses=False, log=False, \
                logdir="./Experiments"):


    distiller = DML(nets, train_loader, None, optimizers,\
                    lr_schedulers=lr_schedulers ,loss_fn=loss_fn,\
                    device=device,log=log, logdir=logdir)
    distiller.train_students(epochs=epochs,plot_losses=plot_losses)
    return distiller.student_cohort


def emsemble_distillate(emsemble, knowledge_net, train_loader, test_loader,stu_optimizer, \
                        lr_schedulers, epochs, loss_fn=nn.KLDivLoss(), \
                        temp=20.0, \
                        distil_weight=1, \
                        device="cpu", \
                        log=False, \
                        logdir="./Experiments", \
                        plot_losses=False):

    distiller = VanillaKD(emsemble, knowledge_net, train_loader, test_loader,\
                           None, stu_optimizer,lr_schedulers,loss_fn=loss_fn,temp=temp,distil_weight=distil_weight, \
                            device=device, log=log, logdir=logdir
                            )
    distiller.train_student(epochs=epochs, plot_losses=plot_losses)
    return distiller.teacher_model, distiller.student_model