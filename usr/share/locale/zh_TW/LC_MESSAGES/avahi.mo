Þ            ß   
        §    h   1  ð     i    b  õ     X     ò       %     5   :     p     ~            "        Á      Ú     û       	   /     9  "   N  4   q  *   ¦  .   Ñ        
             %     7     K     ]     z          §     ¼     Ð     å     û          %     ;     Q     f                    º     Á     Î     í               9     T  %   t  &     #   Á  #   å  #   	  !   -  (   O  <   x  %   µ     Û      ù       #   9     ]     }  #     %   ¹  ?   ß  	        )  %   =     c  
   s     ~                ¸     É     Ü     ê     ú          !     5     C     U  4   m     ¢     ´     Ã     ×     ö               -     E     Z     c     x            '        ¿  &   Ö  	   ý                 #      &      /      @      E      U   L   o   ;   ¼      ø   "   !     :!     G!     U!     b!     p!     t!  	   }!     !  *   !  $   »!  +   à!  #   "  7   0"  %   h"  "   "  4   ±"  (   æ"  (   #     8#     H#     [#     o#     #     #     ¢#     §#     ¸#  
   Ð#  &   Û#  '   $  ,   *$     W$     ]$     r$     v$      $    ³$    L&  g   Û(  ó   C)  e  7*  =  +     Û-     v.     .     .  *   ¼.     ç.     ô.     û.  	   /  "   /     5/      P/     q/     /     /     ª/  '   ½/  .   å/  )   0  .   >0     m0     }0     0     0     ­0     Ä0  $   Ù0     þ0     1     /1     D1     X1     m1     1     1     ­1     Ã1     Ù1     î1     2     &2  "   -2     P2     W2     g2     2     2     ¶2     Ï2  #   è2  $   3  !   13  "   S3  !   v3      3     ¹3  '   Ø3  ;    4  "   <4     _4     {4     4     ¹4     Ø4     ÷4      
5  &   +5  B   R5     5     ¢5     ¶5     Ö5  	   é5     ó5     6     6     26     G6     ]6     m6     }6     6     6     ³6     Ã6     Ù6  +   ï6     7     +7     ;7     T7     j7     z7     7     ¦7     ¿7  	   Õ7     ß7     ò7     ù7     8  '   8     ;8  !   O8  	   q8  	   {8  	   8     8  	   8      8  	   ­8     ·8     Ä8  6   Ý8  *   9     ?9  )   ^9     9     9     ¥9     ²9     Â9  
   Æ9     Ñ9  	   ß9  "   é9     :  !   ':     I:  3   _:     :     ¯:  $   Å:     ê:     ;     ";     5;     C;     Q;     a;     n;     {;     ;     ;     ¬;  '   º;  (   â;  -   <     9<     @<     V<     c<  !   <     C      ,       '                                      r                   T   L   |       \   %   9      e   "         H       A                                 }   j          h         +              #   (                                         k   [   <             D   0   ]   x   t   w   Q   s              z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c             !             
   l   ;      v                        3      p   _       n         u      E       S   W   8   M   N   ^       1      I   *   F   o           b      @   g                        q           2   6   V          X         ?           &       P             i   	             -      =       $                  m         J          G            >   f   ~           4   B           :   Z   `        -h --help            Show this help
    -V --version         Show version
    -D --browse-domains  Browse for browsing domains instead of services
    -a --all             Show all services, regardless of the type
    -d --domain=DOMAIN   The domain to browse in
    -v --verbose         Enable verbose mode
    -t --terminate       Terminate after dumping a more or less complete list
    -c --cache           Terminate after dumping all entries from the cache
    -l --ignore-local    Ignore local services
    -r --resolve         Resolve services found
    -f --no-fail         Don't fail if the daemon is not available
    -p --parsable        Output in parsable format
     -k --no-db-lookup    Don't lookup service types
    -b --dump-db         Dump service type database
 %s [options]

    -h --help            Show this help
    -s --ssh             Browse SSH servers
    -v --vnc             Browse VNC servers
    -S --shell           Browse both SSH and VNC
    -d --domain=DOMAIN   The domain to browse in
 %s [options] %s <host name ...>
%s [options] %s <address ... >

    -h --help            Show this help
    -V --version         Show version
    -n --name            Resolve host name
    -a --address         Resolve address
    -v --verbose         Enable verbose mode
    -6                   Lookup IPv6 address
    -4                   Lookup IPv4 address
 %s [options] %s <name> <type> <port> [<txt ...>]
%s [options] %s <host-name> <address>

    -h --help            Show this help
    -V --version         Show version
    -s --service         Publish service
    -a --address         Publish address
    -v --verbose         Enable verbose mode
    -d --domain=DOMAIN   Domain to publish service in
    -H --host=DOMAIN     Host where service resides
       --subtype=SUBTYPE An additional subtype to register this service with
    -R --no-reverse      Do not publish reverse entry with address
    -f --no-fail         Don't fail if the daemon is not available
 %s [options] <new host name>

    -h --help            Show this help
    -V --version         Show version
    -v --verbose         Enable verbose mode
 : All for now
 : Cache exhausted
 <i>No service currently selected.</i> A NULL terminated list of service types to browse for Access denied Address Address family Address: An unexpected D-Bus error occurred Avahi client failure: %s Avahi domain browser failure: %s Avahi resolver failure: %s Bad number of arguments
 Bad state Browse Service Types Browse service type list is empty! Browsing for service type %s in domain %s failed: %s Browsing for services in domain <b>%s</b>: Browsing for services on <b>local network</b>: Browsing... Canceled.
 Change domain Choose SSH server Choose Shell Server Choose VNC server Client failure, exiting: %s
 Connecting to '%s' ...
 DNS failure: FORMERR DNS failure: NOTAUTH DNS failure: NOTIMP DNS failure: NOTZONE DNS failure: NXDOMAIN DNS failure: NXRRSET DNS failure: REFUSED DNS failure: SERVFAIL DNS failure: YXDOMAIN DNS failure: YXRRSET Daemon connection failed Daemon not running Desktop Disconnected, reconnecting ...
 Domain Domain Name: E Ifce Prot %-*s %-20s Domain
 E Ifce Prot Domain
 Established under name '%s'
 Failed to add address: %s
 Failed to add service: %s
 Failed to add subtype '%s': %s
 Failed to connect to Avahi server: %s Failed to create address resolver: %s
 Failed to create browser for %s: %s Failed to create client object: %s
 Failed to create domain browser: %s Failed to create entry group: %s
 Failed to create host name resolver: %s
 Failed to create resolver for %s of type %s in domain %s: %s Failed to create simple poll object.
 Failed to parse address '%s'
 Failed to parse port number: %s
 Failed to query host name: %s
 Failed to query version string: %s
 Failed to read Avahi domain: %s Failed to register: %s
 Failed to resolve address '%s': %s
 Failed to resolve host name '%s': %s
 Failed to resolve service '%s' of type '%s' in domain '%s': %s
 Host Name Host name conflict
 Host name successfully changed to %s
 Initializing... Interface: Invalid DNS TTL Invalid DNS class Invalid DNS return code Invalid DNS type Invalid Error Code Invalid RDATA Invalid address Invalid argument Invalid configuration Invalid domain name Invalid flags Invalid host name Invalid interface index Invalid number of arguments, expecting exactly one.
 Invalid operation Invalid packet Invalid port number Invalid protocol specification Invalid record Invalid record key Invalid service name Invalid service subtype Invalid service type Is empty Local name collision Location Memory exhausted Name Name collision, picking new name '%s'.
 No command specified.
 No suitable network protocol available Not found Not permitted Not supported OK OS Error Operation failed Port Resolve Service Resolve Service Host Name Resolve the host name of the selected service automatically before returning Resolve the selected service automatically before returning Resource record key is pattern Server version: %s; Host name: %s
 Service Name Service Name: Service Type Service Type: TXT TXT Data TXT Data: Terminal The IP port number of the resolved service The TXT data of the resolved service The address family for host name resolution The address of the resolved service The domain to browse in, or NULL for the default domain The host name of the resolved service The object passed in was not valid The requested operation is invalid because redundant The service name of the selected service The service type of the selected service Timeout reached Too few arguments
 Too many arguments
 Too many clients Too many entries Too many objects Type Version mismatch Waiting for daemon ...
 _Domain... avahi_domain_browser_new() failed: %s
 avahi_service_browser_new() failed: %s
 avahi_service_type_browser_new() failed: %s
 empty execlp() failed: %s
 n/a service_browser failed: %s
 service_type_browser failed: %s
 Project-Id-Version: Avahi
Report-Msgid-Bugs-To: https://github.com/lathiat/avahi/issues
PO-Revision-Date: 2015-08-13 02:16+0000
Last-Translator: Jeff Huang <s8321414@gmail.com>
Language-Team: Chinese (Taiwan) (http://www.transifex.com/lennart/avahi/language/zh_TW/)
Language: zh_TW
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=1; plural=0;
     -h --help            é¡¯ç¤ºæ¬èªªæ
    -V --version         é¡¯ç¤ºçæ¬
    -D --browse-domains  é¡¯ç¤ºé¤äºæåä»¥å¤ççè¦½ç¶²å
    -a --all             é¡¯ç¤ºæææåï¼ç¡è«é¡å
    -d --domain=DOMAIN   çè¦½çç¶²å
    -v --verbose         åç¨åé¤æ¨¡å¼
    -t --terminate       å¨å¾å°åºè¼å®æ´æè¼ä¸å®æ´çåè¡¨å¾çµæ
    -c --cache           å¨å¾å°åºææå¾å¿«åèä¾çé ç®å¾çµæ
    -l --ignore-local    å¿½ç¥æ¬æ©æå
    -r --resolve         è§£æ±ºæ¾å°çæå
    -f --no-fail         è¥å®è­·è¡ç¨ä¸å¯ç¨æä¸è¦å¤±æ
    -p --parsable        ä»¥å¯è§£æçæ ¼å¼è¼¸åº
     -k --no-db-lookup    ä¸è¦æ¥è©¢æåé¡å
    -b --dump-db         å¾å°æåé¡åè³æåº«
 %s [é¸é ]

    -h --help            é¡¯ç¤ºéåèªªæ
    -s --ssh             çè¦½ SSH ä¼ºæå¨
    -v --vnc             çè¦½ VNC ä¼ºæå¨
    -S --shell           åæçè¦½ SSH è VNC
    -d --domain=DOMAIN   è¦çè¦½çåå
 %s [é¸é ] %s <ä¸»æ©åç¨± ...>
%s [é¸é ] %s <å°å ... >

    -h --help            é¡¯ç¤ºæ­¤èª¬æ
    -V --version         é¡¯ç¤ºçæ¬
    -n --name            è§£æ±ºä¸»æ©åç¨±
    -a --address         è§£æ±ºå°å
    -v --verbose         åç¨åé¤æ¨¡å¼
    -6                   æ¥è©¢ IPv6 å°å
    -4                   æ¥è©¢ IPv4 å°å
 %s [é¸é ] %s <åç¨±> <é¡å> <å > [<txt ...>]
%s [é¸é ] %s <ä¸»æ©åç¨±> <å°å>

    -h --help            é¡¯ç¤ºæ­¤èªªæ
    -V --version         é¡¯ç¤ºçæ¬
    -s --service         ç¼ä½æå
    -a --address         ç¼ä½å°å
    -v --verbose         åç¨åé¤æ¨¡å¼
    -d --domain=DOMAIN   è¦ç¼ä½æåçç¶²å
    -H --host=DOMAIN     æåæå¨çä¸»æ©
       --subtype=SUBTYPE è©²æåæè¨»åçé¡å¤å­é¡å
    -R --no-reverse      ä¸è¦ä»¥å°åéåç¼ä½é ç®
    -f --no-fail         è¥å®è­·è¡ç¨ä¸å¯ç¨æä¸è¦å¤±æ
 %s [é¸é ] <æ°ä¸»æ©åç¨±>

    -h --help            é¡¯ç¤ºæ­¤èª¬æ
    -V --version         é¡¯ç¤ºçæ¬
    -v --verbose         åç¨åé¤æ¨¡å¼
 : ç¾å¨æ¯å¨é¨
 : å¿«åå·²èç¡
 <i>ç®åæªé¸åæåã</i> ç©ºççµææåé¡ååè¡¨ï¼å¨çè¦½ æçµå­å ä½å ä½åå®¶æ ä½åï¼ é­éå°æªé æç D-Bus é¯èª¤ Avahi ç¨æ¶ç«¯å¤±æï¼%s Avahi ç¶²åçè¦½å¨å¤±æï¼%s Avahi è§£æå¤±æï¼%s å¼æ¸æ¸ç®ä¸è¯
 çæä¸è¯ çè¦½æåé¡å çè¦½æåé¡åæ¸å®æ¯ç©ºç½çï¼ çè¦½æåé¡å %s å¨ç¶²å %s å¤±æï¼%s æ­£å¨çè¦½ç¶²å <b>%s</b> çæåï¼ æ­£å¨çè¦½<b>æ¬å°ç«¯ç¶²è·¯</b>çæåï¼ æ­£å¨çè¦½... å·²åæ¶ã
 è®æ´ç¶²å é¸æ SSH ä¼ºæå¨ é¸æ Shell ä¼ºæå¨ é¸æ VNC ä¼ºæå¨ ç¨æ¶ç«¯å¤±æï¼æ­£å¨éåºï¼%s
 æ­£å¨é£æ¥å°ã%sã...
 DNS å¤±æï¼FORMERR DNS å¤±æï¼NOTAUTH DNS å¤±æï¼NOTIMP DNS å¤±æï¼NOTZONE DNS å¤±æï¼NXDOMAIN DNS å¤±æï¼NXRRSET DNS å¤±æï¼REFUSED DNS å¤±æï¼SERVFAIL DNS å¤±æï¼YXDOMAIN DNS å¤±æï¼YXRRSET å¹å¾ç¨å¼é£ç·å¤±æ å¹å¾ç¨å¼æ²æå¨å·è¡ä¸­ æ¡é¢ å·²æ·ç·ï¼æ­£å¨éæ°é£æ¥...
 ç¶²å ç¶²ååç¨±ï¼ E Ifce Prot %-*s %-20s ç¶²å
 E Ifce Prot ç¶²å
 å·²æ¼åç¨±ã%sãå»ºç«
 ç¡æ³å å¥ä½åï¼%s
 ç¡æ³å å¥æåï¼%s
 ç¡æ³å å¥å­é¡åã%sãï¼%s
 ç¡æ³é£æ¥è³ Avahi ä¼ºæå¨ï¼%s ç¡æ³å»ºç«ä½åè§£æå¨ï¼%s
 ç¡æ³å»ºç« %s  ççè¦½å¨ï¼%s ç¡æ³å»ºç«å®¢æ¶ç«¯ç©ä»¶ï¼%s
 ç¡æ³å»ºç«ç¶²åçè¦½å¨ï¼%s ç¡æ³å»ºç«æ¢ç®ç¾¤çµï¼%s
 ç¡æ³å»ºç«ä¸»æ©åç¨±è§£æå¨ï¼%s
 å»ºç« %s é¡å %s å¨ç¶²å %s ä¸­çè§£æ³å¨å¤±æï¼%s å»ºåç°¡ææç¥¨ç©ä»¶å¤±æã
 ç¡æ³è§£æä½åã%sã
 ç¡æ³è§£æé£æ¥å èï¼%s
 ç¡æ³æ¥è©¢ä¸»æ©åç¨±ï¼%s
 ç¡æ³æ¥è©¢çæ¬å­ä¸²ï¼%s
 ç¡æ³è®å Avahi ç¶²åï¼%s è¨»åå¤±æï¼%s
 ç¡æ³è§£æä½åã%sãï¼%s
 ç¡æ³è§£æä¸»æ©åç¨±ã%sãï¼%s
 æåã%sãé¡åã%sãå¨ç¶²åã%sãä¸­è§£æ±ºå¤±æï¼%s
 ä¸»æ©åç¨± ä¸»æ©åç¨±è¡çª
 ä¸»æ©åç¨±æåè®æ´çº %s
 æ­£å¨åå§å... ä»é¢ï¼ ç¡æç DNS TTL ç¡æç DNS é¡å¥ ç¡æç DNS åå³ç¢¼ ç¡æç DNS é¡å ç¡æçé¯èª¤ä»£ç¢¼ ç¡æç RDATA ç¡æçä½å ç¡æçå¼æ¸ ç¡æççµæ ç¡æçç¶²ååç¨± ç¡æçææ¨ ç¡æçä¸»æ©åç¨± ç¡æçä»é¢ç´¢å¼ ç¡æçå¼æ¸æ¸ç®ï¼åªé æä¸åã
 ç¡æçæä½ ç¡æçå°å ç¡æçé£æ¥å ç·¨è ç¡æçåå®è¦æ ¼ ç¡æçç´é ç¡æçç´ééé° ç¡æçæååç¨± ç¡æçæåå­é¡å ç¡æçæåé¡å æ¯ç©ºç æ¬æ©åç¨±è¡çª ä½ç½® è¨æ¶é«å·²èç¡ åç¨± åç¨±è¡çªï¼æé¸æ°åç¨± '%s'ã
 æªæå®æä»¤ã
 æ²æåé©çç¶²è·¯åå®å¯ç¨ æ¾ä¸å° æªè¨±å¯ æªæ¯æ´ ç¢ºå® OS é¯èª¤ æä½å¤±æ é£æ¥å  è§£ææå è§£ææåä¸»æ©åç¨± å¨åå³ä¹åèªåè§£ææé¸æåçä¸»æ©åç¨± å¨åå³ä¹åèªåè§£ææé¸çæå è³æºè¨ééé°çºä¸æ¨¡å¼ ä¼ºæå¨çæ¬ï¼%sï¼ä¸»æ©åç¨±ï¼%s
 æååç¨± æååç¨±ï¼ æåé¡å æåé¡åï¼ TXT TXT è³æ TXT è³æï¼ çµç«¯æ© è§£ææåç IP é£æ¥å èç¢¼ è§£ææåç TXT è³æ ä¸»æ©åç¨±è§£æçä½åå®¶æ è§£ææåçä½å è¦çè¦½çç¶²åï¼ææ¯çç©ºçºé è¨­çç¶²å è§£ææåçä¸»æ©åç¨± å³å¥çç©ä»¶ç¡æ è«æ±çæä½ç¡æï¼å çºéè¤ æé¸æåçæååç¨± æé¸æåçæåé¡å å·²éé¾æéå¶ éå°å¼æ¸
 éå¤å¼æ¸
 éå¤ç¨æ¶ç«¯ éå¤æ¢ç® éå¤ç©ä»¶ é¡å çæ¬ä¸ç¬¦ æ­£å¨ç­åå¹å¾ç¨å¼...
 ç¶²å(_D)... avahi_domain_browser_new() å¤±æï¼%s
 avahi_service_browser_new() å¤±æï¼%s
 avahi_service_type_browser_new() å¤±æï¼%s
 ç©ºç execlp() å¤±æï¼%s
 ç¡æ³ä½¿ç¨ service_browser å¤±æï¼%s
 service_type_browser å¤±æï¼%s
 