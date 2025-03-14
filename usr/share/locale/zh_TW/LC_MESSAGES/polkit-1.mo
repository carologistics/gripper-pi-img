��          �   %   �      `  !   a  !   �  #   �     �  ,   �          /  >   6  D   u  ;   �  �   �     �  %   �  #   �     
  $     "   2     U  ,   f  $   �  %   �     �  )   �     &  c  3     �	  �  �	  #   m  "   �  #   �     �  ,   �          9  C   @  N   �  9   �  �        �     �  '   �               8     Q     b  "   ~  %   �  $   �  %   �       3       S                                        	                                                                             
       %s: Argument expected after `%s'
 %s: Invalid --process value `%s'
 %s: Invalid process specifier `%s'
 %s: Subject not specified
 %s: Two arguments expected after `--detail'
 %s: Unexpected argument `%s'
 ACTION Authentication is needed to run `$(program)' as the super user Authentication is needed to run `$(program)' as user $(user.display) Authentication is required to run a program as another user Authentication is required to run the polkit example program Frobnicate (user=$(user), user.gecos=$(user.gecos), user.display=$(user.display), program=$(program), command_line=$(command_line)) BUS_NAME Close FD when the agent is registered Don't replace existing agent if any FD Only output information about ACTION Output detailed action information PID[,START_TIME] Register the agent for the specified process Register the agent owner of BUS_NAME Report bugs to: %s
%s home page: <%s> Run a program as another user Run the polkit example program Frobnicate Show version Usage:
  pkcheck [OPTION...]

Help Options:
  -h, --help                         Show help options

Application Options:
  -a, --action-id=ACTION             Check authorization to perform ACTION
  -u, --allow-user-interaction       Interact with the user if necessary
  -d, --details=KEY VALUE            Add (KEY, VALUE) to information about the action
  --enable-internal-agent            Use an internal authentication agent if necessary
  --list-temp                        List temporary authorizations for current session
  -p, --process=PID[,START_TIME,UID] Check authorization of specified process
  --revoke-temp                      Revoke all temporary authorizations for current session
  -s, --system-bus-name=BUS_NAME     Check authorization of owner of BUS_NAME
  --version                          Show version

Report bugs to: %s
%s home page: <%s>
 [--action-id ACTION] Project-Id-Version: polkit master
Report-Msgid-Bugs-To: https://bugs.freedesktop.org/enter_bug.cgi?product=PolicyKit&keywords=I18N+L10N&component=libpolkit
PO-Revision-Date: 2017-08-31 21:24+0800
Language-Team: Chinese (Taiwan) <chinese-l10n@googlegroups.com>
Language: zh_TW
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Last-Translator: Cheng-Chia Tseng <pswo10680@gmail.com>
X-Generator: Poedit 2.0.3
 %s：預期「%s」後要有引數
 %s：無效 --process 值「%s」
 %s：無效程序指定碼「%s」
 %s：未指定目標
 %s：預期「--detail」後有兩個引數
 %s：未預期引數「%s」
 ACTION 必須先核對身份才能以超級使用者執行「$(program)」 必須先核對身份才能以 $(user.display) 使用者執行「$(program)」 必須先核對身份才能以其他使用者執行程式 必須先核對身份才能執行 polkit 範例程式 Frobnicate (user=$(user), user.gecos=$(user.gecos), user.display=$(user.display), program=$(program), command_line=$(command_line)) BUS_NAME 當代裡已註冊時關閉 FD 不要替換既有代理，若有的話 FD 僅輸出 ACTION 相關資訊 輸出詳細動作資訊 PID[,START_TIME] 為指定程序註冊代理 註冊 BUS_NAME 的代理擁有者 回報臭蟲處：%s
%s 網頁：<%s> 以其他使用者身份執行程式 執行 polkit 範例程式 Frobnicate 顯示版本 用法：
  pkcheck [OPTION...]

幫助選項：
  -h, --help                         顯示幫助選項

應用選項：
  -a, --action-id=ACTION             檢查授權以執行 ACTION
  -u, --allow-user-interaction       若有必要，則和使用者互動
  -d, --details=KEY VALUE            加入 (KEY, VALUE) 到動作的相關資訊中
  --enable-internal-agent            若有必要，使用內部身份核對代理
  --list-temp                        列出目前工作階段的暫時授權
  -p, --process=PID[,START_TIME,UID] 檢查指定程序的授權
  --revoke-temp                      撤銷目前工作階段的所有暫時授權
  -s, --system-bus-name=BUS_NAME     檢查 BUS_NAME 的使用者授權
  --version                          顯示版號

請回報臭蟲到：%s
%s 網頁：<%s>
 [--action-id ACTION] 