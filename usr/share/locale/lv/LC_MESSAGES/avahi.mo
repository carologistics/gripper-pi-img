��    �        �   
      �  �  �  h   1  �   �  i  �  b  �  �   X     �       %     5   :     p     ~     �     �  "   �     �      �     �       	   /     9  "   N  4   q  *   �  .   �        
             %     7     K     ]     z     �     �     �     �     �     �          %     ;     Q     f          �     �     �     �     �     �               9     T  %   t  &   �  #   �  #   �  #   	  !   -  (   O  <   x  %   �     �      �       #   9     ]     }  #   �  %   �  ?   �  	        )  %   =     c  
   s     ~     �     �     �     �     �     �     �          !     5     C     U  4   m     �     �     �     �     �               -     E     Z     c     x     �     �  '   �     �  &   �  	   �                 #      &      /      @      E      U   L   o   ;   �      �   "   !     :!     G!     U!     b!     p!     t!  	   }!     �!  *   �!  $   �!  +   �!  #   "  7   0"  %   h"  "   �"  4   �"  (   �"  (   #     8#     H#     [#     o#     �#     �#     �#     �#     �#  
   �#  &   �#  '   $  ,   *$     W$     ]$     r$     v$      �$  �  �$  �  r&  j   ;)  
  �)  �  �*    @,  �   �.     y/     �/  /   �/  ;   �/     	0     0     0     ,0     40     T0  #   n0     �0     �0     �0     �0  *   �0  9   1  &   Y1  .   �1     �1  	   �1     �1     �1     �1     2     %2     A2     Y2     n2     �2     �2     �2     �2     �2     �2     3     3  !   -3     O3  
   b3  "   m3     �3     �3     �3     �3     �3      �3  !   4  )   >4  +   h4  *   �4  $   �4  (   �4  )   5  '   75  3   _5  9   �5  2   �5      6  $    6  (   E6  '   n6  %   �6     �6  "   �6  ,   �6  >   (7     g7     y7  0   �7     �7  	   �7     �7     �7     8     $8     78     O8     _8     p8     �8     �8     �8     �8     �8  8   9     :9     M9     ^9  "   u9     �9     �9     �9     �9      :  	   :     ":     <:     B:  	   S:  8   ]:     �:  (   �:     �:     �:     �:     ;  
   ;     ;     %;     +;     :;  K   Z;  9   �;  %   �;  *   <     1<     C<     V<     c<     q<     u<  	   ~<     �<  %   �<     �<  (   �<     �<  :   =  "   P=     s=  .   �=     �=  #   �=     >     >     />     H>     ^>     u>     �>     �>     �>     �>  *   �>  +   �>  0   ?     O?     V?     o?     s?  $   �?     C   �   ,       '                                      r      �       �   �   T   L   |       \   %   9      e   "   �   �   H       A   �   �       �   �       �          }   j          h         +   �           #   (               �      �   �                 k   [   <   �   �       D   0   ]   x   t   w   Q   s           �   z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c      �       !       �   �   
   l   ;      v              �          3   �   p   _       n         u      E       S   W   8   M   N   ^       1      I   *   F   o           b   �   @   g              �       �   q           2   6   V          X   �      ?           &       P   �          i   	             -      =       $       �           m   �   �   J          G      �      >   f   ~           4   B           :   Z   `        -h --help            Show this help
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
PO-Revision-Date: 2013-11-20 09:58+0000
Last-Translator: Rūdolfs Mazurs <rudolfs.mazurs@gmail.com>
Language-Team: Latvian (http://www.transifex.com/lennart/avahi/language/lv/)
Language: lv
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n != 0 ? 1 : 2);
     -h --help            Rādīt šo palīdzību
    -V --version         Rādīt versiju
    -D --browse-domains  Pārlūkot pārlūkojamos domēnus, nevis servisus
    -a --all             Rādīt visus servisus, neskatoties uz tipiem
    -d --domain=DOMĒNS   Domēns, kurā pārlūkot
    -v --verbose         Aktivēt detalizētu režīmu
    -t --terminate       Apturēt pēc daudz maz pilnīga saraksta izmešanas
    -c --cache           Apturēt pēc visu ierakstu izmešanas no keša
    -l --ignore-local    Ignorēt lokālos servisus
    -r --resolve         Atrastie meklēšanas servisi
    -f --no-fail         Neavarēt, ja nav atrasts dēmons
    -p --parsable        Izvade parsējamā formā
     -k --no-db-lookup    Neuzmeklēt servisu tipus
    -b --dump-db         Izmest servisu tipu datubāzi
 %s [opcijas]

    -h --help            Rādīt šo palīdzību
    -s --ssh             Pārlūkot SSH serverus
    -v --vnc             Pārlūkot VNC serverus
    -S --shell           Pārlūkot gan SSH, gan VNC
    -d --domain=DOMĒNS   Domēns, kurā pārlūkot
 %s [opcijas] %s <datora nosaukums...>
%s [opcijas] %s <adrese ... >

    -h --help            Rādīt šo palīdzību
    -V --version         Rādīt versiju
    -n --name            Atrast datora nosaukumu
    -a --address         Atrast adresi
    -v --verbose         Aktivēt detalizēto režīmi
    -6                   Uzmeklēt IPv6 adreses
    -4                   Uzmeklēt IPv4 adreses
 %s [opcijas] %s <nosaukums> <tips> <ports> [<txt ...>]
%s [opcijas] %s <datora-nosaukums> <adrese>

    -h --help            Rādīt šo palīdzību
    -V --version         Rādīt versiju
    -s --service         Publicēt servisu
    -a --address         Publicēt adresi
    -v --verbose         Aktivēt detalizēto režīmu
    -d --domain=DOMĒNS   Domēns, kurā publicēt servisu
    -H --host=DOMĒNS     Kur atrodas serviss
       --subtype=APAKŠTIPS Papildu apakštips, ar ko reģistrēt šo servisu
    -R --no-reverse      Nepublicēt apgriezto ierakstu ar adresi
    -f --no-fail         Neavarēt, ja dēmons nav pieejams
 %s [opcijas] <jauns datora nosaukums>

    -h --help            Rādīt šo palīdzību
    -V --version         Rādīt versiju
    -v --verbose         Aktivēt detalizēto režīmi
 : Pagaidām viss
 : Kešs izsmelts
 <i>Neviens serviss pašlaik nav izvēlēts.</i> Ar NULL pabeigts saraksts ar servisu veidiem, ko pārlūkot Pieeja liegta Adrese Adrešu saime Adrese: Notika negaidīta D-Bus kļūda Avahi klienta kļūme: %s Avahi domēna pārlūka kļūme: %s Avahi atradēja kļūme: %s Slikts parametru saraksts
 Slikts stāvoklis Pārlūkot servisa tipus Pārlūka servisa tipu saraksts ir tukšs! Servisa tipa %s pārlūkošana domēnā %s neizdevās: %s Pārlūkot servisus domēnā<b>%s</b>: Pārlūkot servisus <b>lokālajā tīklā</b>: Pārlūko... Atcelts.
 Mainīt domēnu Izvēlieties SSH serveri Izvēlieties čaulas serveri Izvēlieties VNC serveri Klienta kļūme, iziet: %s
 Savienojas ar '%s' ...
 DNS kļūme: FORMERR DNS kļūme: NOTAUTH DNS kļūme: NOTIMP DNS kļūme: NOTZONE DNS kļūme: NXDOMAIN DNS kļūme: NXRRSET DNS kļūme: REFUSED DNS kļūme: SERVFAIL DNS kļūme: YXDOMAIN DNS kļūme: YXRRSET Neizdevās savienoties ar dēmonu Dēmons nedarbojas Darbvirsma Atvienojās, atkal savienojas ...
 Domēns Domēna nosaukums: E Ifce Prot %-*s %-20s domēns
 E Ifce Prot domēns
 Izveido ar nosaukumu '%s'
 Neizdevās pievienot adresi: %s
 Neizdevās pievienot servisu: %s
 Neizdevās pievienot apakštipu '%s': %s
 Neizdevās savienoties ar Avahi serveri: %s Neizdevās izveidot adrešu atradēju: %s
 Neizdevās izveidot %s pārlūku: %s Neizdevās izveidot klienta objektu: %s
 Neizdevās izveidot domēna pārlūku: %s Neizdevās izveidot ierakstu grupu: %s
 Neizdevās izveidot datoru nosaukumu atradēju: %s
 Neizdevās izveidot %s atradēju tipam %s domēnā %s: %s Neizdevās izveidot vienkāršu aptaujas objektu.
 Neizdevās parsēt adresi '%s'
 Neizdevās parsēt porta numuru: %s
 Neizdevās vaicāt datora nosaukumu: %s
 Neizdevās vaicāt versijas virkni: %s
 Neizdevās nolasīt Avahi domēnu: %s Neizdevās reģistrēt: %s
 Neizdevās atrast adresi '%s': %s
 Neizdevās atrast datora nosaukumu '%s': %s
 Neizdevās atrast servisu '%s' ar tipu '%s' domēnā '%s': %s
 Servera nosaukums Datoru nosaukumu konflikts
 Datora nosaukums ir veiksmīgi nomainīts uz %s
 Inicializē... Saskarne: Nederīgs DNS TTL Nederīga DNS klase Nederīgs DNS atgrieztais kods Nederīgs DNS tips Nederīgs kļūdas kods Nederīgs RDATA Nederīga adrese Nederīgs parametrs Nederīga konfigurācija Nederīgs domēna nosaukums Nederīgs karogs Nederīgs datora nosaukums Nederīgs saskarnes indekss Nederīgs parametru skaits, tiek gaidīts tieši viens.
 Nederīga darbība Nederīga pakete Nederīgs porta numurs Nederīga protokola specifikācija Nederīgs ieraksts Nederīga ieraksta atslēga Nederīgs servisa nosaukums Nederīgs servisa apakštips Nederīgs servisa veids Ir tukšs Lokālā vārda kolīzija Vieta Atmiņa izsmelta Nosaukums Nosaukumu kolīzija, izvēlieties jaunu nosaukumu '%s'.
 Nav norādīta komanda.
 Nav pieejams piemērots tīkla protokols Nav atrasts Nav atļauts Nav atbalstīts Labi OS kļūda Darbība neizdevās Ports Atrast servisu Atrast servisa datora nosaukumu Pirms atgriezties, atrast izvēlētā servisa datora nosaukumu automātiski Pirms atgriezties, atrast izvēlēto servisu automātiski Resursa ieraksta atslēga ir šablons Servera versija: %s; Datora nosaukums: %s
 Servisa nosaukums Servisa nosaukums: Servisa tips Servisa tips: TXT TXT dari TXT dati: Terminālis IP porta skaitlis atrastajam servisam Atrastā servisa TXT dati Adrešu saime datora nosaukuma atradumam Atrastā servisa adrese Domēns, kurā pārlūkot, vai NULL noklusētajam domēnam Atrastā servisa servera nosaukums Padotais objekts nav derīgs Pieprasītā darbība nav derīga, jo ir lieka Izvēlētā servisa nosaukums Servisa tips izvēlētajam servisam Tika sasniegta noildze Pārāk maz parametru
 Pārāk daudz parametru
 Pārāk daudz klientu Pārāk daudz ierakstu Pārāk daudz objektu Tips Versiju nesakritība Gaida uz dēmonu ...
 _Domēns... avahi_domain_browser_new() neizdevās: %s
 avahi_service_browser_new() neizdevās: %s
 avahi_service_type_browser_new() neizdevās: %s
 tukšs execlp() neizdevās: %s
 n/p service_browser neizdevās: %s
 service_type_browser neizdevās: %s
 