from django.apps import AppConfig


class FormConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'titanik_form'

    def ready(self):
        import titanik_form.dash_apps.finished_apps.titanik
