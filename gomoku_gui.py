import pygame
import sys
import math
from Gomoku import Gomoku


class GomokuGUI:
    def __init__(self, size: int = 15):
        pygame.init()
        
        # Game state
        self.screen_state = "menu"  # "menu" or "game"
        self.board_size = size
        self.game = None
        
        self.cell_size = 40
        self.margin = 30
        
        # Animation variables
        self.animating = False
        self.anim_stone = None
        self.anim_row = 0
        self.anim_col = 0
        self.anim_player = 1
        self.anim_progress = 0
        self.anim_duration = 0.5  # seconds
        self.hover_pos = None
        
        # Win animation variables
        self.winning_line = None
        self.win_anim_progress = 0
        self.show_win_popup = False
        self.popup_main_rect = None
        self.popup_rematch_rect = None
        self.popup_hover = None
        
        # Menu variables
        self.menu_buttons = []
        self.size_buttons = []
        self.selected_size = 15
        
        # Colors
        self.BOARD_COLOR = (218, 165, 32)
        self.LINE_COLOR = (139, 69, 19)
        self.BLACK_STONE = (44, 62, 80)
        self.WHITE_STONE = (255, 255, 255)
        self.BG_COLOR = (44, 62, 80)
        self.TEXT_COLOR = (255, 255, 255)
        self.BUTTON_COLOR = (39, 174, 96)
        self.BUTTON_HOVER = (46, 204, 113)
        self.WIN_LINE_COLOR = (255, 215, 0)
        self.POPUP_BG = (44, 62, 80, 230)
        
        # Window setup
        self.width = 700
        self.height = 700
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku Game")
        
        self.title_font = pygame.font.Font(None, 80)
        self.font = pygame.font.Font(None, 36)
        self.button_font = pygame.font.Font(None, 28)
        
        self.button_rect = None
        self.button_hover = False
        
        self.setup_menu_buttons()
    
    def setup_menu_buttons(self):
        # Play button
        self.play_button = pygame.Rect(self.width // 2 - 100, 350, 200, 60)
        
        # Size selection buttons
        sizes = [(9, "Easy"), (15, "Medium"), (19, "Hard")]
        button_width = 120
        button_height = 50
        total_width = len(sizes) * button_width + (len(sizes) - 1) * 20
        start_x = (self.width - total_width) // 2
        
        self.size_buttons = []
        for i, (size, label) in enumerate(sizes):
            x = start_x + i * (button_width + 20)
            rect = pygame.Rect(x, 480, button_width, button_height)
            self.size_buttons.append((rect, size, label))
    
    def start_game(self):
        self.screen_state = "game"
        self.game = Gomoku(size=self.selected_size)
        
        # Recalculate window size for game
        board_size = self.cell_size * self.selected_size + 2 * self.margin
        self.width = board_size
        self.height = board_size + 120
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        # Setup game button
        self.button_rect = pygame.Rect(self.width // 2 - 80, board_size + 20, 160, 40)
        
        # Reset animations
        self.animating = False
        self.winning_line = None
        self.win_anim_progress = 0
        self.show_win_popup = False
        self.popup_main_rect = None
        self.popup_rematch_rect = None
        self.popup_hover = None

    def reset_current_game(self):
        if not self.game:
            return
        self.game.make_init()
        self.animating = False
        self.winning_line = None
        self.win_anim_progress = 0
        self.show_win_popup = False
        self.popup_main_rect = None
        self.popup_rematch_rect = None
        self.popup_hover = None

    def go_to_menu(self):
        self.screen_state = "menu"
        self.game = None
        self.width = 700
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.setup_menu_buttons()
        self.button_rect = None
        self.button_hover = False
        self.animating = False
        self.winning_line = None
        self.win_anim_progress = 0
        self.show_win_popup = False
        self.popup_main_rect = None
        self.popup_rematch_rect = None
        self.popup_hover = None
    
    def draw_menu(self):
        self.screen.fill(self.BG_COLOR)
        
        # Title with gradient effect
        title_text = "GOMOKU"
        title_surface = self.title_font.render(title_text, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 150))
        
        # Shadow
        shadow_surface = self.title_font.render(title_text, True, (0, 0, 0))
        shadow_rect = shadow_surface.get_rect(center=(self.width // 2 + 3, 153))
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle = "Five in a Row"
        subtitle_surface = self.font.render(subtitle, True, (189, 195, 199))
        subtitle_rect = subtitle_surface.get_rect(center=(self.width // 2, 220))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Play button
        button_color = self.BUTTON_HOVER if self.button_hover else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, button_color, self.play_button, border_radius=10)
        pygame.draw.rect(self.screen, (255, 215, 0), self.play_button, 3, border_radius=10)
        
        play_text = self.font.render("PLAY", True, self.TEXT_COLOR)
        play_rect = play_text.get_rect(center=self.play_button.center)
        self.screen.blit(play_text, play_rect)
        
        # Board size label
        size_label = self.button_font.render("Select Difficulty:", True, self.TEXT_COLOR)
        size_label_rect = size_label.get_rect(center=(self.width // 2, 440))
        self.screen.blit(size_label, size_label_rect)
        
        # Size buttons
        for rect, size, label in self.size_buttons:
            is_selected = size == self.selected_size
            is_hovered = rect.collidepoint(pygame.mouse.get_pos())
            
            if is_selected:
                color = (255, 215, 0)
                border_color = (255, 215, 0)
                border_width = 4
            elif is_hovered:
                color = (70, 70, 70)
                border_color = (189, 195, 199)
                border_width = 2
            else:
                color = (60, 60, 60)
                border_color = (100, 100, 100)
                border_width = 2
            
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_color, rect, border_width, border_radius=8)
            
            # Label
            label_surface = self.button_font.render(label, True, self.TEXT_COLOR)
            label_rect = label_surface.get_rect(center=(rect.centerx, rect.centery - 8))
            self.screen.blit(label_surface, label_rect)
            
            # Size
            size_text = f"{size}x{size}"
            size_surface = pygame.font.Font(None, 20).render(size_text, True, (150, 150, 150))
            size_rect = size_surface.get_rect(center=(rect.centerx, rect.centery + 12))
            self.screen.blit(size_surface, size_rect)
        
        # Instructions
        instructions = ["Five stones in a row wins!", "Click to place your stone"]
        for i, text in enumerate(instructions):
            inst_surface = pygame.font.Font(None, 24).render(text, True, (120, 120, 120))
            inst_rect = inst_surface.get_rect(center=(self.width // 2, 590 + i * 30))
            self.screen.blit(inst_surface, inst_rect)
        
    def draw_board(self):
        if not self.game:
            return
        
        # Background
        self.screen.fill(self.BG_COLOR)
        
        # Board background
        board_rect = pygame.Rect(0, 60, self.width, self.width)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)
        
        # Grid lines
        for i in range(self.game.size):
            x = self.margin + i * self.cell_size
            y = self.margin + 60 + i * self.cell_size
            
            # Vertical line
            pygame.draw.line(
                self.screen, self.LINE_COLOR,
                (x, self.margin + 60),
                (x, self.margin + 60 + (self.game.size - 1) * self.cell_size),
                2
            )
            # Horizontal line
            pygame.draw.line(
                self.screen, self.LINE_COLOR,
                (self.margin, y),
                (self.margin + (self.game.size - 1) * self.cell_size, y),
                2
            )
        
        # Star points (for 15x15 board)
        if self.game.size == 15:
            star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
            for r, c in star_points:
                x = self.margin + c * self.cell_size
                y = self.margin + 60 + r * self.cell_size
                pygame.draw.circle(self.screen, self.LINE_COLOR, (x, y), 4)
        
        # Draw stones
        for r in range(self.game.size):
            for c in range(self.game.size):
                if self.game.board[r][c] != 0:
                    # Don't draw the stone being animated
                    if self.animating and r == self.anim_row and c == self.anim_col:
                        continue
                    self.draw_stone(r, c, self.game.board[r][c], 1.0)
        
        # Draw hover effect
        if self.hover_pos and self.game.status() is None and not self.animating:
            row, col = self.hover_pos
            if 0 <= row < self.game.size and 0 <= col < self.game.size:
                if self.game.board[row][col] == 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + 60 + row * self.cell_size
                    radius = self.cell_size // 2 - 3
                    color = self.BLACK_STONE if self.game.to_move == 1 else self.WHITE_STONE
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*color, 80), (radius, radius), radius)
                    self.screen.blit(s, (x - radius, y - radius))
        
        # Draw winning line
        if self.winning_line:
            self.draw_winning_line()
    
    def draw_stone(self, row, col, player, alpha=1.0):
        x = self.margin + col * self.cell_size
        y = self.margin + 60 + row * self.cell_size
        radius = self.cell_size // 2 - 3
        
        color = self.BLACK_STONE if player == 1 else self.WHITE_STONE
        
        # Create surface with alpha for fade effect
        s = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        
        # Draw stone with alpha
        stone_color = (*color, int(255 * alpha))
        pygame.draw.circle(s, stone_color, (radius + 2, radius + 2), radius)
        
        # Border
        border_color = (52, 73, 94) if player == 1 else (189, 195, 199)
        border_alpha = (*border_color, int(255 * alpha))
        pygame.draw.circle(s, border_alpha, (radius + 2, radius + 2), radius, 2)
        
        self.screen.blit(s, (x - radius - 2, y - radius - 2))
    
    def draw_animated_stone(self, dt):
        if not self.animating:
            return
        
        self.anim_progress += dt / self.anim_duration
        
        if self.anim_progress >= 1.0:
            self.anim_progress = 1.0
            self.animating = False
        
        # Easing function (ease out bounce)
        t = self.anim_progress
        if t < 0.5:
            # Falling
            ease = t * t * 2
        else:
            # Bounce
            t_bounce = (t - 0.5) * 2
            ease = 0.5 + 0.5 * (1 - (1 - t_bounce) ** 3)
        
        # Calculate position
        start_y = -50
        end_y = self.margin + 60 + self.anim_row * self.cell_size
        current_y = start_y + (end_y - start_y) * ease
        
        x = self.margin + self.anim_col * self.cell_size
        radius = self.cell_size // 2 - 3
        
        # Add squash and stretch effect on impact
        scale_x = 1.0
        scale_y = 1.0
        if t > 0.8:
            impact_t = (t - 0.8) / 0.2
            scale_y = 1.0 - 0.2 * math.sin(impact_t * math.pi)
            scale_x = 1.0 + 0.1 * math.sin(impact_t * math.pi)
        
        color = self.BLACK_STONE if self.anim_player == 1 else self.WHITE_STONE
        
        # Draw stone with scaling
        scaled_radius_x = int(radius * scale_x)
        scaled_radius_y = int(radius * scale_y)
        
        s = pygame.Surface((scaled_radius_x * 2 + 4, scaled_radius_y * 2 + 4), pygame.SRCALPHA)
        pygame.draw.ellipse(s, color, (2, 2, scaled_radius_x * 2, scaled_radius_y * 2))
        
        border_color = (52, 73, 94) if self.anim_player == 1 else (189, 195, 199)
        pygame.draw.ellipse(s, border_color, (2, 2, scaled_radius_x * 2, scaled_radius_y * 2), 2)
        
        self.screen.blit(s, (x - scaled_radius_x - 2, int(current_y) - scaled_radius_y - 2))
    
    def find_winning_line(self):
        """Find the five stones that form the winning line."""
        result = self.game.status()
        if result not in [1, -1]:
            return None
        
        for r in range(self.game.size):
            for c in range(self.game.size):
                if self.game.board[r][c] == result:
                    # Check all directions
                    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    for dr, dc in directions:
                        line = [(r, c)]
                        # Forward
                        nr, nc = r + dr, c + dc
                        while (0 <= nr < self.game.size and 0 <= nc < self.game.size and 
                               self.game.board[nr][nc] == result):
                            line.append((nr, nc))
                            nr += dr
                            nc += dc
                        # Backward
                        nr, nc = r - dr, c - dc
                        while (0 <= nr < self.game.size and 0 <= nc < self.game.size and 
                               self.game.board[nr][nc] == result):
                            line.insert(0, (nr, nc))
                            nr -= dr
                            nc -= dc
                        
                        if len(line) >= 5:
                            return line[:5]
        return None
    
    def draw_winning_line(self):
        """Draw animated line over winning stones."""
        if not self.winning_line or len(self.winning_line) < 2:
            return
        
        # Animate line appearance
        alpha = min(255, int(self.win_anim_progress * 255))
        
        start_r, start_c = self.winning_line[0]
        end_r, end_c = self.winning_line[-1]
        
        start_x = self.margin + start_c * self.cell_size
        start_y = self.margin + 60 + start_r * self.cell_size
        end_x = self.margin + end_c * self.cell_size
        end_y = self.margin + 60 + end_r * self.cell_size
        
        # Draw glowing line
        for width in [12, 8, 4]:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            line_alpha = alpha // (width // 4 + 1)
            pygame.draw.line(s, (*self.WIN_LINE_COLOR, line_alpha), 
                           (start_x, start_y), (end_x, end_y), width)
            self.screen.blit(s, (0, 0))
        
        # Pulsing circles on winning stones
        pulse = (math.sin(self.win_anim_progress * 8) + 1) / 2
        for r, c in self.winning_line:
            x = self.margin + c * self.cell_size
            y = self.margin + 60 + r * self.cell_size
            radius = int((self.cell_size // 2 - 3) * (1 + pulse * 0.3))
            s = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
            circle_alpha = int(alpha * 0.5)
            pygame.draw.circle(s, (*self.WIN_LINE_COLOR, circle_alpha), 
                             (radius + 5, radius + 5), radius, 3)
            self.screen.blit(s, (x - radius - 5, y - radius - 5))
    
    def draw_ui(self):
        if not self.game:
            return
        
        # Status text
        if self.game.status() is None:
            player_name = "Player 1 (Black)" if self.game.to_move == 1 else "Player 2 (White)"
            status_text = f"{player_name}'s Turn"
        elif self.game.status() == 1:
            status_text = "Player 1 (Black) Wins! 🎉"
        elif self.game.status() == -1:
            status_text = "Player 2 (White) Wins! 🎉"
        else:
            status_text = "Draw!"
        
        text_surface = self.font.render(status_text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(self.width // 2, 30))
        self.screen.blit(text_surface, text_rect)
        
        # New Game button
        button_color = self.BUTTON_HOVER if self.button_hover else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, button_color, self.button_rect, border_radius=5)
        
        button_text = self.button_font.render("New Game", True, self.TEXT_COLOR)
        button_text_rect = button_text.get_rect(center=self.button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        # Draw win popup
        if self.show_win_popup:
            self.draw_win_popup()
    
    def handle_click(self, pos):
        x, y = pos
        
        if self.screen_state == "menu":
            # Check play button
            if self.play_button.collidepoint(pos):
                self.start_game()
                return
            
            # Check size buttons
            for rect, size, label in self.size_buttons:
                if rect.collidepoint(pos):
                    self.selected_size = size
                    return
        
        elif self.screen_state == "game":
            # Check button click
            if self.button_rect and self.button_rect.collidepoint(pos):
                self.reset_current_game()
                return

            if self.show_win_popup:
                if self.popup_main_rect and self.popup_main_rect.collidepoint(pos):
                    self.go_to_menu()
                    return
                if self.popup_rematch_rect and self.popup_rematch_rect.collidepoint(pos):
                    self.reset_current_game()
                    return
                return
            
            # Check board click
            if y < 60 or self.animating:
                return
            
            if self.game.status() is not None:
                return
            
            x -= self.margin
            y -= self.margin + 60
            
            col = round(x / self.cell_size)
            row = round(y / self.cell_size)
            
            if not (0 <= row < self.game.size and 0 <= col < self.game.size):
                return
            
            try:
                # Start animation
                self.anim_row = row
                self.anim_col = col
                self.anim_player = self.game.to_move
                self.anim_progress = 0
                self.animating = True
                
                self.game.make_move((row, col))
                
                # Check for win and start win animation
                if self.game.status() is not None:
                    self.winning_line = self.find_winning_line()
                    self.win_anim_progress = 0
                    self.show_win_popup = False
            except ValueError:
                pass
    
    def draw_win_popup(self):
        """Draw victory popup overlay."""
        result = self.game.status()
        if result is None:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill(self.POPUP_BG)
        self.screen.blit(overlay, (0, 0))
        
        # Popup box
        popup_width = 420
        popup_height = 260
        popup_x = (self.width - popup_width) // 2
        popup_y = (self.height - popup_height) // 2
        
        # Animated scale
        scale = min(1.0, self.win_anim_progress * 2)
        scaled_width = int(popup_width * scale)
        scaled_height = int(popup_height * scale)
        scaled_x = popup_x + (popup_width - scaled_width) // 2
        scaled_y = popup_y + (popup_height - scaled_height) // 2
        
        popup_rect = pygame.Rect(scaled_x, scaled_y, scaled_width, scaled_height)
        pygame.draw.rect(self.screen, (52, 73, 94), popup_rect, border_radius=20)
        pygame.draw.rect(self.screen, self.WIN_LINE_COLOR, popup_rect, 4, border_radius=20)
        
        if scale > 0.5:  # Show text after popup appears
            # Title
            if result == 1:
                title = "Player 1 Wins!"
            elif result == -1:
                title = "Player 2 Wins!"
            else:
                title = "Draw!"
            
            title_font = pygame.font.Font(None, 48)
            title_surface = title_font.render(title, True, self.WIN_LINE_COLOR)
            title_rect = title_surface.get_rect(center=(self.width // 2, popup_y + 60))
            self.screen.blit(title_surface, title_rect)

            # Buttons
            button_width = 160
            button_height = 50
            gap = 20
            total_width = button_width * 2 + gap
            start_x = self.width // 2 - total_width // 2
            btn_y = popup_y + 140

            self.popup_main_rect = pygame.Rect(start_x, btn_y, button_width, button_height)
            self.popup_rematch_rect = pygame.Rect(start_x + button_width + gap, btn_y, button_width, button_height)

            main_color = self.BUTTON_HOVER if self.popup_hover == "main" else self.BUTTON_COLOR
            rematch_color = self.BUTTON_HOVER if self.popup_hover == "rematch" else (52, 152, 219)

            pygame.draw.rect(self.screen, main_color, self.popup_main_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.WIN_LINE_COLOR, self.popup_main_rect, 2, border_radius=8)
            pygame.draw.rect(self.screen, rematch_color, self.popup_rematch_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.WIN_LINE_COLOR, self.popup_rematch_rect, 2, border_radius=8)

            main_text = self.button_font.render("Main Menu", True, self.TEXT_COLOR)
            main_rect = main_text.get_rect(center=self.popup_main_rect.center)
            self.screen.blit(main_text, main_rect)

            rematch_text = self.button_font.render("Rematch", True, self.TEXT_COLOR)
            rematch_rect = rematch_text.get_rect(center=self.popup_rematch_rect.center)
            self.screen.blit(rematch_text, rematch_rect)
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.screen_state == "menu":
                        self.button_hover = self.play_button.collidepoint(event.pos)
                    elif self.screen_state == "game" and self.button_rect:
                        self.button_hover = self.button_rect.collidepoint(event.pos)

                        if self.show_win_popup:
                            if self.popup_main_rect and self.popup_main_rect.collidepoint(event.pos):
                                self.popup_hover = "main"
                            elif self.popup_rematch_rect and self.popup_rematch_rect.collidepoint(event.pos):
                                self.popup_hover = "rematch"
                            else:
                                self.popup_hover = None
                        else:
                            self.popup_hover = None
                            
                            # Calculate hover position on board
                            x, y = event.pos
                            if self.game and y >= 60 and not self.animating:
                                x -= self.margin
                                y -= self.margin + 60
                                col = round(x / self.cell_size)
                                row = round(y / self.cell_size)
                                if 0 <= row < self.game.size and 0 <= col < self.game.size:
                                    self.hover_pos = (row, col)
                                else:
                                    self.hover_pos = None
                            else:
                                self.hover_pos = None
            
            if self.screen_state == "menu":
                self.draw_menu()
            elif self.screen_state == "game":
                # Update win animation
                if self.winning_line and self.win_anim_progress < 2.0:
                    self.win_anim_progress += dt * 1.5
                    if self.win_anim_progress >= 0.8 and not self.show_win_popup:
                        self.show_win_popup = True
                
                self.draw_board()
                self.draw_animated_stone(dt)
                self.draw_ui()
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = GomokuGUI(size=15)
    gui.run()
