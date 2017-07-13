from subprocess import Popen, PIPE
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

class EmailNotification:

    @staticmethod
    def sendEmail(text, subject=None):
        sender = 'jean-martin.albert@gameloft.com'
        recipients = ['jm.albert@gmail.com']
        subject = subject
        msg = MIMEMultipart()
        msg.preamble = subject
        msg.add_header("From", sender)
        msg.add_header("Subject", subject)
        msg.add_header("To", ", ".join(recipients))
        msg.attach(MIMEText(text, 'html'))

        try:
            p = Popen(["/usr/sbin/sendmail", "-t"], stdin=PIPE)
            p.communicate(msg.as_string().encode('utf-8'))
        except Exception as e:
            print('ERROR', e)

#    @staticmethod
#    def getTraceback():
#        trcbk = traceback.format_exc()
#        return trcbk
#
#    @staticmethod
#    def emailTraceback(subject='Programmatic Traceback'):
#        tb = ExceptionManager.getTraceback()
#        ExceptionManager.sendEmail(tb, subject)
if __name__ == '__main__':
    EmailNotification.sendEmail("This is a test")
