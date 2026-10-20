# -*- coding: utf-8 -*-
# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import sys
import multiprocessing
from contextlib import redirect_stdout
from io import StringIO
from PySide6.QtCore import Qt
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QApplication
# qfluentwidgets 1.7.7 currently prints a commercial-product promotion while
# importing its config module. Suppress only that third-party import-time output;
# project logs and runtime errors keep their normal stdout behavior.
with redirect_stdout(StringIO()):
    from qfluentwidgets import (FluentWindow, FluentIcon,NavigationItemPosition, InfoBar)

from backend.config import config, tr
from backend.tools.process_manager import ProcessManager
from ui.advanced_setting_interface import AdvancedSettingInterface
from ui.enhance_interface import EnhanceInterface
from ui.home_interface import HomeInterface
from ui.image_format_interface import ImageFormatInterface


# 项目内置图标由用户指定的 PNG 转换而来，避免运行时依赖用户目录中的绝对路径。
PROJECT_ICON_PATH = "design/clean-canvas.ico"


class CleanCanvasWindow(FluentWindow):
    """应用外壳：负责页面注册、窗口状态和进程级资源清理。

    各业务页面自行维护任务与预览状态；此类只负责把它们放入 Fluent
    导航框架，避免把页面业务逻辑混入窗口生命周期代码。
    """

    def __init__(self):
        super().__init__()
        # 内容修复是以视频预览为核心的桌面工作台。低于该尺寸时，右侧设置、
        # 任务与预览会同时失去可读性；以参考版常用 1360px 宽度作为最小基线，
        # 而不是在过窄窗口中将预览压成一条窄栏。
        self.setMinimumSize(1360, 720)
        # 禁用云母效果
        self.setMicaEffectEnabled(False)
        # 设置深色主题并跟随系统主题色
        # setTheme(Theme.LIGHT)
        # setThemeColor(getSystemAccentColor(), save=True)

        # 初始化系统主题监听器并连接信号
        # self.themeListener = SystemThemeListener(self)
        # self.themeListener.start()

        # 同时设置应用与窗口图标，使系统标题栏、任务栏缩略图和帮助窗口保持一致。
        icon = QtGui.QIcon(PROJECT_ICON_PATH)
        QtWidgets.QApplication.instance().setWindowIcon(icon)
        self.setWindowIcon(icon)
        # 系统标题栏不能只把字符串的一部分缩小。开发版本改在“关于”卡片中以
        # 辅助字号展示，标题栏只保留产品名，避免版本号喧宾夺主。
        self.setWindowTitle(tr['CleanCanvasWindow']['Title'])
        # 创建界面布局
        self._create_layout()
        self._connectSignalToSlot()
        self._lazy_check_update()

    def _lazy_check_update(self):
        """ 延迟检查更新 """
        if not config.checkUpdateOnStartup.value:
            return
        self.check_update_timer = QtCore.QTimer(self)
        self.check_update_timer.setSingleShot(True)
        self.check_update_timer.timeout.connect(lambda: self.advancedSettingInterface.check_update(ignore=True))
        self.check_update_timer.start(2000)

    def _connectSignalToSlot(self):
        config.appRestartSig.connect(self._showRestartTooltip)

    def _showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            'Updated successfully',
            'Configuration takes effect after restart',
            duration=5000,
            parent=self
        )

    def _create_layout(self):
        # 页面实例在此统一持有。导航路由由 addSubInterface 自动建立，页面间
        # 不通过切换 stack index 来交换业务状态。
        self.homeInterface = HomeInterface(self)
        self.homeInterface.setObjectName("HomeInterface")
        self.advancedSettingInterface = AdvancedSettingInterface(self)
        self.advancedSettingInterface.setObjectName("AdvancedSettingInterface")
        self.enhanceInterface = EnhanceInterface(self)
        self.enhanceInterface.setObjectName("EnhanceInterface")
        self.imageFormatInterface = ImageFormatInterface(self)
        self.imageFormatInterface.setObjectName("ImageFormatInterface")

        # 宽窗口中沿用原版的内嵌展开导航：侧栏展开会让内容区域重新布局，而
        # 不是以浮层覆盖预览。内容页已移除过大的最小宽度，因此展开/收起不会
        # 再把顶层窗口永久撑大；较窄窗口仍由 QFluentWidgets 自动使用菜单浮层。
        self.navigationInterface.setMinimumExpandWidth(1360)

        # 添加到主窗口作为子界面
        # 这是内容修复工作台，而不是应用“首页”；使用画笔图标能更准确表达
        # 去文字、掩膜编辑和图像修复的核心功能。
        self.addSubInterface(self.homeInterface, FluentIcon.BRUSH, tr['CleanCanvasWindow']['Workspace'])
        self.addSubInterface(self.enhanceInterface, FluentIcon.PHOTO, "画质增强")
        self.addSubInterface(self.imageFormatInterface, FluentIcon.SAVE_AS, "图片转换")
        # 高级设置属于低频全局操作，放在底部能保持工作台导航的视觉重心。
        # 页面只在这里注册一次，路由键由 AdvancedSettingInterface 唯一确定。
        self.addSubInterface(
            self.advancedSettingInterface,
            FluentIcon.SETTING,
            tr['Setting']['AdvancedSetting'],
            NavigationItemPosition.BOTTOM,
        )

    def on_navigation_item_changed(self, key):
        """导航项变更时的处理函数"""
        if key == 'main':
            self.stackWidget.setCurrentIndex(0)
        elif key == 'advanced':
            self.stackWidget.setCurrentIndex(1)

    def closeEvent(self, event):
        """保存窗口几何，并终止仍由应用登记的子进程后退出。"""
        self.save_window_position()
        # 界面退出时，清理子进程
        ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

    def save_window_position(self):
        """保存窗口位置到配置文件"""
        # 保存窗口位置和大小
        config.set(config.windowX, self.x())
        config.set(config.windowY, self.y())
        config.set(config.windowW, self.width())
        config.set(config.windowH, self.height())

    def update_progress(self):
        # 定时器轮询更新进度（现在更新到视频滑块上）
        if self.se is not None:
            try:
                pos = min(self.frame_count - 1, int(self.se.progress_total / 100 * self.frame_count))
                if pos != self.video_slider.value():
                    self.video_slider.setValue(pos)
                # 检查是否完成
                if self.se.isFinished:
                    self.processing_finished()
            except Exception as e:
                # 捕获任何异常，防止崩溃
                print(f"更新进度时出错: {str(e)}")

    def load_window_position(self):
        # 尝试读取窗口位置
        try:
            x = config.windowX.value
            y = config.windowY.value
            width = config.windowW.value
            height = config.windowH.value

            if not x or not y:
                self.center_window()
                return

            # 确保窗口在屏幕内
            screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
            if (x >= 0 and y >= 0 and
                x + width <= screen_rect.width() and
                y + height <= screen_rect.height()):
                self.setGeometry(x, y, width, height)
            else:
                self.center_window()
        except Exception as e:
            print(e)
            self.center_window()

    def center_window(self):
        """将窗口居中显示"""
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        window_rect = self.frameGeometry()
        center_point = screen_rect.center()
        window_rect.moveCenter(center_point)
        self.move(window_rect.topLeft())

if __name__ == '__main__':
    # 以 spawn 的方式启动子进程
    # 该进程会启动一个新的 python 解释器执行
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    window = CleanCanvasWindow()
    # 先设置透明, 再显示, 否则会有闪烁的效果
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()
    # 使用动画效果逐渐显示窗口
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)  # 300毫秒的动画
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    app.exec()
