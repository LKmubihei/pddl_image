(define (problem picture_112)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_battery red_pump red_pump_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_battery)
        (clear red_pump)
        (clear red_pump_1)
        (part_at red_pump_1 table)
        (part_at red_battery buffer_placement)
        (part_at blue_battery battery_placement)
        (part_at red_pump pump_placement)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
